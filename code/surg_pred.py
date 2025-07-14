import pandas as pd
import numpy as np

# Modeling & preprocessing imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline as SKPipeline
from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import optuna
from sklearn.metrics import classification_report, confusion_matrix

# Explainability
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------------------
# 1. Load data
# ----------------------------------------
df = pd.read_csv('data/df_updated_V05.csv')

# ----------------------------------------
# 2. Recoding raw columns
# ----------------------------------------
to_integer = [
    "marital_status", "stage", "seq_num", "age", "race", "Sex",
    "State", "County", "Dual_Elig", "CCI", "RUCC", "AIDS_HIV"
]
for col in to_integer:
    df[col] = df[col].fillna(999).astype(int)

# Unique cancer-type count
cancer_counts = df.groupby("patient_id")["cancer_type"] \
                  .nunique() \
                  .rename("unique_cancer_types") \
                  .reset_index()
df = df.merge(cancer_counts, on="patient_id", how="left")

# Simple recodes
df["race_recode"] = (df["race"] == 1).astype(int)
df["SVI_HH_COMP_recode"] = (df["SVI_HH_COMP"] < 0.3).astype(int)

# County recode for small-count buckets
cnt = df["County"].value_counts()
df["County_recode"] = df["County"].where(cnt >= 100, 999)
cnt2 = df["County_recode"].value_counts()
df["County_recode_2"] = df["County_recode"].where(cnt2 >= 300, 998)

# Stage recode function
def map_stage(x):
    if x == 0: return 0
    if x == 1: return 1
    if x in {2,3,4,5}: return 2
    if x == 7: return 3
    return 4

df["stage_recode"] = df["stage"].apply(map_stage)

# Filter for valid surgery refusal labels and demographics
df = df[df["Surgery_refusal"].isin([0,1])]
df = df[df["age_group"] != 1]            # drop <65
df = df[df["State"] != 6]               # drop specific state code
df = df[~df["cancer_type"].isin(["oropharynx","penis"])]

# Additional recodes
df["CCI_recode"] = df["CCI"].map({0:0,1:1}).fillna(2).astype(int)
df["marital_recode"] = (df["marital_status"] == 2).astype(int)
df["cancer_type_recode"] = df["cancer_type"].map({
    "vulva":0, "anus":1, "cervix":2, "oropharynx":3, "penis":4
}).fillna(5).astype(int)

# ----------------------------------------
# 3. Define features & target
# ----------------------------------------
features = [
    "marital_recode","stage_recode","age_group","race_recode","Sex",
    "Dual_Elig","CCI_recode","RUCC","County_recode_2","cancer_type_recode",
    "seq_num","unique_cancer_types","SVI_SOCIECO","SVI_HH_COMP_recode",
    "SVI_MINO","SVI_HH_TRANS","SURG_SPECS_RATE",
    # comorbidities
    "myocardial_infraction","congestive_heart_failure","peripheral_vascular",
    "cerebrovascular","dementia","chronic_pulmonary","rheumatic",
    "peptic_ulcer","mild_liver","diabetes","diabetes_comlication",
    "hemiplegia","renal","malignancy","severe_liver","metastatic_solid_tumor",
    "AIDS_HIV"
]
target = "Surgery_refusal"

X = df[features]
y = df[target].astype(int)

# ----------------------------------------
# 4. Train/test split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.4, random_state=42
)

# ----------------------------------------
# 5. Preprocessing pipelines
# ----------------------------------------
numeric_cols = [
    "seq_num","unique_cancer_types",
    "SVI_SOCIECO","SVI_MINO","SVI_HH_TRANS","SURG_SPECS_RATE"
]  # continuous
categorical_cols = [c for c in features if c not in numeric_cols]

numeric_transformer = SKPipeline([
    ("imputer", KNNImputer(n_neighbors=5)),
    ("scaler", StandardScaler())
])
categorical_transformer = SKPipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# ----------------------------------------
# 6. Pipeline with SMOTETomek & classifier
# ----------------------------------------
resampler = SMOTETomek(random_state=42)
base_clf = XGBClassifier(
    tree_method="hist", use_label_encoder=False,
    eval_metric="logloss", random_state=42
)
pipeline = IMBPipeline([
    ("pre", preprocessor),
    ("resample", resampler),
    ("clf", base_clf)
])

# ----------------------------------------
# 7. Hyperparameter tuning via Optuna
# ----------------------------------------
def objective(trial):
    params = {
        "clf__n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
        "clf__max_depth": trial.suggest_int("max_depth", 3, 10),
        "clf__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "clf__subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "clf__colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "clf__gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "clf__lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "clf__alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
    }
    pipeline.set_params(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, timeout=1800)

print("Best hyperparameters:", study.best_params)

# ----------------------------------------
# 8. Final training & evaluation
# ----------------------------------------
best_params = {k.replace("clf__", ""): v for k, v in study.best_params.items()}
final_clf = XGBClassifier(
    tree_method="hist", use_label_encoder=False,
    eval_metric="logloss", random_state=42, **best_params
)
pipeline.set_params(clf=final_clf)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------------------
# 9. SHAP explainability
# ----------------------------------------
X_test_pre = pipeline.named_steps["pre"].transform(X_test)
feature_names = pipeline.named_steps["pre"].get_feature_names_out()

explainer = shap.TreeExplainer(pipeline.named_steps["clf"])
shap_values = explainer.shap_values(X_test_pre)

mpl.rcParams["font.family"] = "serif"
shap.summary_plot(shap_values, X_test_pre, feature_names=feature_names, max_display=20)

shap_int = explainer.shap_interaction_values(X_test_pre)
shap.summary_plot(shap_int, X_test_pre, feature_names=feature_names, max_display=20)
