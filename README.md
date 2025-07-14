# XGBoost‑SHAP Modeling of Treatment Refusal in SEER–Medicare HPV-Associated Cancer Cohort

**Original code prepared by:** Maryam Kheirandish, PhD, Emory University (@Marykheirandish-dev) 📝

**Revised and organized by:** Ryan Suk, PhD, Emory University (Principal Investigator)

**Data Source:** SEER–Medicare linked database 🏥


---

## Project Overview 🌟

This repository focuses on predicting two key refusal behaviors in HPV-associated cancer patients:

1. **Radiotherapy (RT) refusal** 💥
2. **Surgery refusal** 🔪

Using SEER–Medicare data, we:

* Preprocess registry and claims variables
* Train XGBoost classifiers to predict refusal outcomes
* Analyze feature importance and interactions via SHAP

---
## >> 🤖 To check our PRELIMINARY RESULTS >>> 👉[Click here to view the preliminary results](https://github.com/ryan-suk/SHAP_HPV_Tx_Refusal/tree/main/prelim)
---
## Code Organization 📂

```
  data/     # Datasets
  code/     # All scripts (recoding, modeling, tuning)
  prelim/  # Generated outputs (figures, tables)
  README.md # Project overview and instructions
```

---

## Dependencies 🔌

````

Key Python libraries:
- `pandas`, `numpy` 
- `scikit-learn` 
- `xgboost` 
- `shap` 
- `imbalanced-learn` 
- `optuna` 
- `matplotlib` 
````
---

## Usage 🏃‍♀️

* Conducts train/test split
* Imputes missing data
* One-hot encodes categorical features
* Performs SMOTETomek resampling
* Tunes hyperparameters via Optuna
* Trains final XGBoost model and reports accuracy, confusion matrix, classification report

**Explainability with SHAP**:

   * Generates SHAP summary plot (top 20 features)
   * Computes SHAP interaction values for key feature pairs

**Results**:
   Check the `reports/` folder for saved SHAP figures and model performance summaries.

---

## Contact 📬

For questions or collaboration, please reach out to Dr. Maryam Kheirandish at **[mkheira@emory.edu](mailto:mkheira@emory.edu)**.

Project lead: Dr. Ryan Suk, Emory University **[ryan.suk@emory.edu](mailto:ryan.suk@emory.edu)** 🌐 https://www.ryansuk.com/research-team

---

*This work adheres to SEER–Medicare data use agreements.*
