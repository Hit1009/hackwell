# AI-Driven Risk Prediction Engine for Chronic Care Patients

## Overview

This project implements an **AI-driven risk prediction engine** that forecasts whether a chronic care patient is at risk of deterioration in the next 90 days. It uses patient data (demographics, vitals, lab results, medication adherence, lifestyle logs) to generate risk scores, explain predictions, and present results in a clinician-friendly dashboard.


---

## Features

* **Synthetic Multi-Disease Dataset**

  * Patients simulated across 5 chronic diseases: Diabetes, Obesity, Heart Failure, CKD, and COPD
  * Two linked datasets:

    * `patients.csv`: demographics + aggregated features + risk label
    * `timeseries.csv`: daily vitals/labs over 30–180 days for trend visualization

* **Prediction Model**

  * Gradient Boosting classifier (scikit-learn)
  * Input: 30–180 days of patient data (summarized into features)
  * Output: probability of deterioration in the next 90 days

* **Evaluation Metrics**

  * AUROC, AUPRC, Accuracy, Precision, Recall, F1-score
  * Calibration plot + Brier score
  * Confusion matrix

* **Explainability**

  * **Global:** SHAP summary plots for population-level feature importance
  * **Local:** SHAP force plots + text-based explanations for patient-level predictions
  * Clinician-friendly rules for “Recommended Action”

* **Dashboard Prototype (Colab/Streamlit ready)**

  * **Cohort View:** risk scores for all patients, sorted high to low
  * **Patient Detail View:** time-series trends, key feature drivers, next-action suggestion

---

## Project Structure

```
hackwell_new.py        # Main notebook/script
patients.csv           # Aggregated patient-level dataset
timeseries.csv         # Daily vitals & labs for each patient
README.md              # Documentation
```

---

## Installation & Requirements

Clone the repo or upload the notebook to Colab.

### Dependencies

* Python 3.8+
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* shap

Install missing packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

---

## How to Run

1. Train the model on `patients.csv`.
2. Evaluate metrics (AUROC, AUPRC, calibration, etc.).
3. Use the **Cohort View** to see risk scores across patients.
4. Pick a patient ID → view **Patient Detail View** with:

   * Risk probability
   * Time-series plots (BP, Glucose, eGFR, etc.)
   * Local SHAP explanation
   * Recommended action

---

## Example Outputs

* **Cohort View:** table of patients sorted by risk probability
* **Patient Detail:**

  * Blood pressure and glucose trends
  * SHAP plot explaining top features
  * Text-based driver summary

---

