import pandas as pd
import numpy as np

# Modeling & preprocessing imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline as SKPipeline
from imblearn.pipeline import Pipeline as IMBPipe
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import optuna
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Explainability
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------
# 1. Load & split data
# ---------------------------
df = pd.read_csv('df_updated_V05.csv')  # replace with your path

# Define features and target
features = [
    'marital_recode', 'stage_recode', 'age_group', 'race', 'Sex',
    'Dual_Elig', 'MAdvantage_plan', 'State', 'County', 'cancer_type_recode',
    'seq_num_recode', 'YOST', 'CCI', 'RUCC', 'diabetes', 'AIDS_HIV',
    'SVI_SOCIECO', 'SVI_HH_COMP', 'SVI_MINO', 'SVI_HH_TRANS',
    'unique_cancer_types', 'RADI_RATE'
]
target = 'RT_refusal'

X = df[features]
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)

# ---------------------------
# 2. Preprocessing setup
# ---------------------------
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = [c for c in features if c not in numeric_cols]

numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# ---------------------------
# 3. Pipeline with resampling
# ---------------------------
resampler = SMOTETomek(random_state=42)

# Placeholder classifier; will be set in Optuna objective
clf = XGBClassifier(tree_method='hist', use_label_encoder=False, eval_metric='logloss', random_state=42)

pipeline = IMBPipe(steps=[
    ('pre', preprocessor),
    ('resample', resampler),
    ('clf', clf)
])

# ---------------------------
# 4. Hyperparameter tuning with Optuna
# ---------------------------
def objective(trial):
    params = {
        'clf__n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
        'clf__max_depth': trial.suggest_int('max_depth', 3, 10),
        'clf__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'clf__subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'clf__colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'clf__gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'clf__lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'clf__alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
    }
    pipeline.set_params(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=1800)

print("Best hyperparameters:", study.best_params)

# ---------------------------
# 5. Train final model
# ---------------------------
best_params = {k.replace('clf__', ''): v for k, v in study.best_params.items()}
final_clf = XGBClassifier(tree_method='hist', use_label_encoder=False,
                          eval_metric='logloss', random_state=42, **best_params)
pipeline.set_params(clf=final_clf)
pipeline.fit(X_train, y_train)

# ---------------------------
# 6. Evaluate on test set
# ---------------------------
y_pred = pipeline.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# 7. SHAP Explainability
# ---------------------------
# Extract preprocessed test features
X_test_trans = pipeline.named_steps['pre'].transform(X_test)
feature_names = pipeline.named_steps['pre'].get_feature_names_out()

explainer = shap.TreeExplainer(pipeline.named_steps['clf'])
shap_values = explainer.shap_values(X_test_trans)

# Summary plot
mpl.rcParams['font.family'] = 'serif'
shap.summary_plot(shap_values, X_test_trans, feature_names=feature_names, max_display=20)

# Interaction summary (top 20)
shap_interactions = explainer.shap_interaction_values(X_test_trans)
shap.summary_plot(shap_interactions, X_test_trans, feature_names=feature_names, max_display=20)
