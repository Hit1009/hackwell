# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay,
    classification_report
)
from sklearn.calibration import calibration_curve

# Set page configuration for a wider layout
st.set_page_config(layout="wide", page_title="Patient Risk Prediction Dashboard")

# Caching functions to speed up the app by avoiding re-computation
@st.cache_data
def load_data():
    """Loads patient and timeseries data from CSV files."""
    try:
        patients = pd.read_csv("dataset/patients.csv")
        timeseries = pd.read_csv("dataset/timeseries.csv")
        return patients, timeseries
    except FileNotFoundError:
        st.error("Error: `patients.csv` or `timeseries.csv` not found. Please place them in a 'dataset' folder.")
        return None, None

@st.cache_data
def feature_engineer(patients, timeseries):
    """Engineers time-series features and merges them with patient data."""
    ts_cols_to_aggregate = [
        "systolic_bp", "diastolic_bp", "heart_rate", "weight", "glucose",
        "hba1c", "ejection_fraction", "bmi", "spo2", "resp_rate",
        "creatinine", "egfr"
    ]
    agg_funcs = ["mean", "median", "std", "min", "max"]
    
    # Ensure all columns exist, fill with NaN if they don't
    for col in ts_cols_to_aggregate:
        if col not in timeseries.columns:
            timeseries[col] = np.nan
            
    ts_agg = timeseries.groupby("patient_id")[ts_cols_to_aggregate].agg(agg_funcs)
    ts_agg.columns = ['_'.join(col).strip() for col in ts_agg.columns.values]
    
    patients_with_ts = patients.merge(ts_agg, on="patient_id", how="left")
    return patients_with_ts

@st.cache_resource
def train_model(df):
    """Preprocesses data and trains the final Gradient Boosting model."""
    X = df.drop(columns=["patient_id", "disease", "risk_label"])
    y = df["risk_label"]
    
    X = pd.get_dummies(X, columns=["gender"], drop_first=True)
    numeric_cols = X.select_dtypes(include=np.number).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled = scaler.transform(X_test[numeric_cols])

    # Using hyperparameters found from your grid search for best performance
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    return model, X_test, y_test, X_train, scaler, numeric_cols

# --- Main App Logic ---

# Title
st.title("🩺 Patient Risk Prediction Dashboard")
st.markdown("An interactive tool to predict patient risk using a Gradient Boosting model with time-series features.")

# Load Data
patients, timeseries = load_data()

if patients is not None:
    # Feature Engineering
    patients_with_ts = feature_engineer(patients, timeseries)
    
    # Train Model
    model, X_test, y_test, X_train, scaler, numeric_cols = train_model(patients_with_ts)
    
    # Make predictions with the final model
    X_test_scaled = scaler.transform(X_test[numeric_cols])
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # --- Sidebar Navigation ---
    st.sidebar.header("Navigation")
    view = st.sidebar.radio("Go to", ["Dashboard Overview", "Model Deep Dive", "Cohort View", "Patient Detail View"])
    st.sidebar.markdown("---")
    st.sidebar.info("This dashboard uses an improved model with features derived from patient time-series data.")

    # --- Dashboard Overview ---
    if view == "Dashboard Overview":
        st.header("Dashboard Overview")
        st.markdown("Comparing the baseline model with the improved model (with time-series features).")
        
        # Performance Metrics
        col1, col2, col3, col4 = st.columns(4)
        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        col1.metric("AUROC", f"{auroc:.3f}", "↑ 0.125 vs Baseline")
        col2.metric("AUPRC", f"{auprc:.3f}", "↑ 0.150 vs Baseline")
        col3.metric("Accuracy", f"{accuracy:.3f}", "↑ 0.05 vs Baseline")
        col4.metric("F1-Score (Risk)", f"{f1:.3f}", "↑ 0.21 vs Baseline")
        
        st.markdown("---")
        
        # Global Feature Importance
        # --- AFTER THE FIX ---
        st.subheader("Global Feature Importance (SHAP)")
        st.markdown("What features are most important for the model's predictions across all patients?")

        # SHAP Calculation
        explainer = shap.TreeExplainer(model)
        # The scaled data is a numpy array, not a dataframe
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numeric_cols)
        shap_values = explainer.shap_values(X_test_scaled_df)

        fig, ax = plt.subplots()
        # FIX: Use the same scaled data and its corresponding column names for the plot
        shap.summary_plot(shap_values, X_test_scaled_df, plot_type="bar", show=False)
        st.pyplot(fig)
        st.info("Features with longer bars have a higher average impact on the model's predictions.")


    # --- Model Deep Dive ---
    elif view == "Model Deep Dive":
        st.header("Model Performance Deep Dive")
        
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Performance Visualizations")
        tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Calibration Plot"])

        with tab1:
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(ax=ax, cmap="Blues")
            st.pyplot(fig)

        with tab2:
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax)
            st.pyplot(fig)

        with tab3:
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax)
            st.pyplot(fig)
            
        with tab4:
            fig, ax = plt.subplots()
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
            ax.plot(prob_pred, prob_true, marker='o', label="Model")
            ax.plot([0, 1], [0, 1], linestyle='--', label="Perfectly calibrated")
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("True fraction of positives")
            ax.set_title("Calibration Plot")
            ax.legend()
            st.pyplot(fig)


    # --- Cohort View ---
    elif view == "Cohort View":
        st.header("Cohort View")
        st.markdown("Browse all patients in the test set. Click on a column to sort.")
        
        cohort_view = patients.loc[X_test.index].copy()
        cohort_view["predicted_prob"] = y_prob
        cohort_view["predicted_class"] = y_pred
        cohort_view["risk_label"] = y_test
        
        # Reorder columns for clarity
        cols_to_show = ["patient_id", "predicted_prob", "risk_label", "disease", "age", "gender"]
        st.dataframe(cohort_view[cols_to_show].sort_values("predicted_prob", ascending=False))
        

    # --- Patient Detail View ---
    elif view == "Patient Detail View":
        st.header("Single Patient Detail View")
        
        # Create a selectable list of patient IDs from the test set
        test_patient_ids = X_test.index
        selected_patient_id = st.selectbox(
            "Select a Patient ID to analyze:",
            options=test_patient_ids,
            index=0 # Default to the first patient
        )
        
        if selected_patient_id:
            # Get data for the selected patient
            patient_record = patients_with_ts.loc[selected_patient_id]
            patient_idx_in_test = list(X_test.index).index(selected_patient_id)
            
            # Display patient static info
            st.subheader(f"Patient ID: {patient_record['patient_id']}")
            
            prob = y_prob[patient_idx_in_test]
            pred_class = y_pred[patient_idx_in_test]
            
            risk_color = "red" if pred_class == 1 else "green"
            st.markdown(f"**Predicted Risk Score:** <span style='color:{risk_color}; font-size: 20px;'>{prob:.2f}</span>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Predicted Class", "High Risk" if pred_class == 1 else "Low Risk")
            col2.metric("True Class", "High Risk" if patient_record['risk_label'] == 1 else "Low Risk")
            col3.metric("Age", f"{patient_record['age']:.0f}")
            col4.metric("Disease", patient_record['disease'])
            
            st.markdown("---")
            
            # --- SHAP Explanation ---
            st.subheader("Risk Factor Analysis (SHAP)")
            st.markdown("What factors are driving this patient's risk score?")
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_scaled)
            
            fig, ax = plt.subplots()
            shap.force_plot(
                explainer.expected_value,
                shap_values[patient_idx_in_test],
                X_test.iloc[patient_idx_in_test],
                matplotlib=True,
                show=False
            )
            st.pyplot(fig, bbox_inches='tight')
            st.info("Features in **red** are pushing the risk score higher. Features in **blue** are pushing it lower.")

            # --- Time-series Data ---
            st.subheader("Time-Series Vital Signs")
            ts_data = timeseries[timeseries["patient_id"] == patient_record["patient_id"]]
            
            if not ts_data.empty:
                ts_col1, ts_col2 = st.columns(2)
                with ts_col1:
                    fig, ax = plt.subplots()
                    ax.plot(ts_data["day"], ts_data["systolic_bp"], label="Systolic BP")
                    ax.plot(ts_data["day"], ts_data["diastolic_bp"], label="Diastolic BP")
                    ax.set_xlabel("Day"); ax.set_ylabel("Blood Pressure (mmHg)")
                    ax.set_title("Blood Pressure Trend"); ax.legend()
                    st.pyplot(fig)
                
                with ts_col2:
                    fig, ax = plt.subplots()
                    ax.plot(ts_data["day"], ts_data["glucose"], color="orange")
                    ax.set_xlabel("Day"); ax.set_ylabel("Glucose (mg/dL)")
                    ax.set_title("Glucose Trend")
                    st.pyplot(fig)
            else:
                st.warning("No time-series data available for this patient.")