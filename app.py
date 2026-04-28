"""
frontend/app.py
───────────────
Streamlit frontend for the Multi-Disease Prediction System.

Run:
    cd frontend
    streamlit run app.py
"""

import os
import sys
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediPredict – Multi-Disease Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(99,179,237,0.2);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
}
.main-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #e2e8f0;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0.5rem 0 0 0;
}
.card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.result-positive {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.5rem 0;
}
.result-negative {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #22c55e;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.5rem 0;
}
.risk-low    { color: #4ade80; font-weight: 700; font-size: 1.1rem; }
.risk-medium { color: #fbbf24; font-weight: 700; font-size: 1.1rem; }
.risk-high   { color: #f87171; font-weight: 700; font-size: 1.1rem; }
.metric-box {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #63b3ed;
}
.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #e2e8f0;
    border-bottom: 2px solid #1e3a5f;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
.stTabs [data-baseweb="tab"] {
    font-size: 1rem !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 MediPredict</h1>
    <p>AI-powered Multi-Disease Risk Assessment · Powered by XGBoost · Diabetes · Heart Disease · Liver Disease</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_url = st.text_input("API URL", value=API_URL)

    st.markdown("---")
    st.markdown("### 🔬 Select Diseases to Predict")
    predict_diabetes = st.checkbox("🩺 Diabetes",       value=True)
    predict_heart    = st.checkbox("❤️ Heart Disease",  value=True)
    predict_liver    = st.checkbox("🫀 Liver Disease",  value=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app uses **XGBoost** classifiers trained on clinical datasets to predict disease risk.
    
    **Disclaimer:** This tool is for educational purposes only. Always consult a licensed physician.
    """)

    # API status check
    st.markdown("---")
    st.markdown("### 📡 API Status")
    try:
        r = requests.get(f"{api_url}/", timeout=2)
        if r.status_code == 200:
            data = r.json()
            if data.get("models_loaded"):
                st.success("✅ API Online · Models Loaded")
            else:
                st.warning("⚠️ API Online · Models NOT Loaded\nRun train_models.py first")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline\nStart backend with:\n`uvicorn main:app --reload`")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_viz, tab_about = st.tabs(
    ["🔮 Predict", "📊 Visualizations", "📋 Documentation"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    payload = {}

    # ── DIABETES INPUTS ───────────────────────────────────────────────────────
    if predict_diabetes:
        st.markdown('<p class="section-title">🩺 Diabetes Risk Factors</p>', unsafe_allow_html=True)
        with st.expander("Enter Diabetes Parameters", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                preg    = st.number_input("Pregnancies",        min_value=0,   max_value=20,  value=2,   step=1)
                glucose = st.number_input("Glucose (mg/dL)",    min_value=0,   max_value=300, value=120, step=1)
            with c2:
                bp      = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=180, value=72, step=1)
                skin    = st.number_input("Skin Thickness (mm)", min_value=0,  max_value=100, value=20,  step=1)
            with c3:
                insulin = st.number_input("Insulin (mu U/ml)",  min_value=0,   max_value=900, value=80,  step=1)
                bmi     = st.number_input("BMI",                 min_value=0.0, max_value=80.0,value=28.0,step=0.1)
            with c4:
                dpf     = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, step=0.01)
                age_d   = st.number_input("Age (years)",         min_value=1,   max_value=120, value=33,  step=1)

        payload["diabetes"] = {
            "Pregnancies": preg, "Glucose": glucose, "BloodPressure": bp,
            "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age_d,
        }

    # ── HEART DISEASE INPUTS ──────────────────────────────────────────────────
    if predict_heart:
        st.markdown('<p class="section-title">❤️ Heart Disease Risk Factors</p>', unsafe_allow_html=True)
        with st.expander("Enter Heart Disease Parameters", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                h_age    = st.number_input("Age",    min_value=1,   max_value=120, value=52, step=1, key="h_age")
                h_sex    = st.selectbox("Sex",  ["male", "female"])
                h_cp     = st.selectbox("Chest Pain Type",
                    ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
            with c2:
                h_trest  = st.number_input("Resting BP (mmHg)", min_value=80, max_value=250, value=130)
                h_chol   = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=230)
                h_fbs    = st.selectbox("Fasting Blood Sugar >120", [0, 1], format_func=lambda x: "Yes" if x else "No")
            with c3:
                h_ecg    = st.selectbox("Resting ECG", [0, 1, 2])
                h_thal   = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=155, key="h_thalmax")
                h_exang  = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "Yes" if x else "No")
            with c4:
                h_old    = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
                h_slope  = st.selectbox("ST Slope", [0, 1, 2])
                h_ca     = st.selectbox("Vessels (Fluoroscopy)", [0, 1, 2, 3, 4])
            h_thal_type = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

        payload["heart"] = {
            "age": h_age, "sex": h_sex, "cp": h_cp, "trestbps": h_trest,
            "chol": h_chol, "fbs": h_fbs, "restecg": h_ecg, "thalach": h_thal,
            "exang": h_exang, "oldpeak": h_old, "slope": h_slope, "ca": h_ca,
            "thal": h_thal_type,
        }

    # ── LIVER DISEASE INPUTS ──────────────────────────────────────────────────
    if predict_liver:
        st.markdown('<p class="section-title">🫀 Liver Disease Risk Factors</p>', unsafe_allow_html=True)
        with st.expander("Enter Liver Disease Parameters", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                l_age    = st.number_input("Age",    min_value=1,  max_value=120, value=45, step=1, key="l_age")
                l_gender = st.selectbox("Gender", ["Male", "Female"])
                l_tbil   = st.number_input("Total Bilirubin",   min_value=0.0, max_value=80.0, value=1.0, step=0.1)
            with c2:
                l_dbil   = st.number_input("Direct Bilirubin",  min_value=0.0, max_value=40.0, value=0.3, step=0.1)
                l_alkp   = st.number_input("Alkaline Phosphotase", min_value=0, max_value=2000, value=200, step=1)
            with c3:
                l_alt    = st.number_input("ALT (Alamine Aminotransferase)",  min_value=0, max_value=2000, value=25, step=1)
                l_ast    = st.number_input("AST (Aspartate Aminotransferase)", min_value=0, max_value=5000, value=30, step=1)
            with c4:
                l_prot   = st.number_input("Total Proteins (g/dL)", min_value=0.0, max_value=15.0, value=6.8, step=0.1)
                l_alb    = st.number_input("Albumin (g/dL)",         min_value=0.0, max_value=10.0, value=3.5, step=0.1)
                l_agr    = st.number_input("Albumin/Globulin Ratio", min_value=0.0, max_value=5.0,  value=1.0, step=0.01)

        payload["liver"] = {
            "Age": l_age, "Gender": l_gender, "Total_Bilirubin": l_tbil,
            "Direct_Bilirubin": l_dbil, "Alkaline_Phosphotase": l_alkp,
            "Alamine_Aminotransferase": l_alt, "Aspartate_Aminotransferase": l_ast,
            "Total_Protiens": l_prot, "Albumin": l_alb, "Albumin_and_Globulin_Ratio": l_agr,
        }

    # ── SUBMIT ────────────────────────────────────────────────────────────────
    st.markdown("---")
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_predict = st.button("🔮 Run Prediction", type="primary", use_container_width=True)
    with col_info:
        st.info("💡 Fill in the patient parameters above and click **Run Prediction** to get results.")

    if run_predict:
        if not payload:
            st.error("Please select at least one disease to predict.")
        else:
            with st.spinner("Analyzing patient data..."):
                try:
                    # Remove keys not selected
                    payload_clean = {k: v for k, v in payload.items()
                                     if (k == "diabetes" and predict_diabetes) or
                                        (k == "heart"    and predict_heart)    or
                                        (k == "liver"    and predict_liver)}

                    response = requests.post(
                        f"{api_url}/predict",
                        json=payload_clean,
                        timeout=10,
                    )

                    if response.status_code == 200:
                        results = response.json()["predictions"]
                        st.success("✅ Prediction complete!")

                        disease_names = {
                            "diabetes": "🩺 Diabetes",
                            "heart":    "❤️ Heart Disease",
                            "liver":    "🫀 Liver Disease",
                        }
                        disease_descs = {
                            "diabetes": "Type 2 Diabetes Mellitus",
                            "heart":    "Coronary Heart Disease",
                            "liver":    "Liver Cirrhosis / Hepatitis",
                        }

                        st.markdown("## 📋 Prediction Results")
                        cols = st.columns(len(results))

                        for col, (disease, res) in zip(cols, results.items()):
                            with col:
                                is_positive = res["prediction"] == 1
                                css_class   = "result-positive" if is_positive else "result-negative"
                                icon        = "⚠️" if is_positive else "✅"
                                prob        = res["probability"]
                                risk        = res["risk_level"]

                                risk_class = {
                                    "Low":    "risk-low",
                                    "Medium": "risk-medium",
                                    "High":   "risk-high",
                                }[risk]

                                st.markdown(f"""
                                <div class="{css_class}">
                                    <h3 style="color:#e2e8f0;margin:0">{disease_names[disease]}</h3>
                                    <p style="color:#94a3b8;margin:0.2rem 0 0.8rem 0;font-size:0.85rem">{disease_descs[disease]}</p>
                                    <div style="font-size:2.5rem;margin:0.5rem 0">{icon}</div>
                                    <div style="font-size:1.3rem;font-weight:700;color:#e2e8f0">{res['result']}</div>
                                    <div style="font-size:0.9rem;color:#94a3b8;margin-top:0.5rem">
                                        Probability: <span style="color:#63b3ed;font-weight:600">{prob:.1f}%</span>
                                    </div>
                                    <div style="margin-top:0.4rem">
                                        Risk: <span class="{risk_class}">{risk}</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                        # Probability bar chart
                        st.markdown("---")
                        st.markdown("### 📊 Risk Probability Overview")
                        fig, ax = plt.subplots(figsize=(10, 3))
                        fig.patch.set_facecolor("#0f172a")
                        ax.set_facecolor("#1e293b")

                        diseases = list(results.keys())
                        probs    = [results[d]["probability"] for d in diseases]
                        labels   = [disease_names[d] for d in diseases]
                        colours  = ["#ef4444" if p >= 65 else "#fbbf24" if p >= 35 else "#4ade80"
                                    for p in probs]

                        bars = ax.barh(labels, probs, color=colours, height=0.5, edgecolor="#334155")
                        ax.axvline(35, color="#fbbf24", linestyle="--", alpha=0.5, linewidth=1)
                        ax.axvline(65, color="#ef4444", linestyle="--", alpha=0.5, linewidth=1)

                        for bar, prob in zip(bars, probs):
                            ax.text(min(prob + 1, 95), bar.get_y() + bar.get_height()/2,
                                    f"{prob:.1f}%", va="center", color="white", fontsize=11, fontweight="bold")

                        ax.set_xlim(0, 105)
                        ax.set_xlabel("Risk Probability (%)", color="#94a3b8")
                        ax.tick_params(colors="#94a3b8")
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        for spine in ["left", "bottom"]:
                            ax.spines[spine].set_color("#334155")

                        low_patch    = mpatches.Patch(color="#4ade80", label="Low Risk (<35%)")
                        medium_patch = mpatches.Patch(color="#fbbf24", label="Medium Risk (35–65%)")
                        high_patch   = mpatches.Patch(color="#ef4444", label="High Risk (>65%)")
                        ax.legend(handles=[low_patch, medium_patch, high_patch],
                                  facecolor="#1e293b", labelcolor="white", fontsize=9, loc="lower right")

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                    elif response.status_code == 503:
                        st.error("🔴 Models not loaded. Please run `notebooks/train_models.py` first.")
                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("""
                    ❌ **Cannot connect to the backend API.**
                    
                    Please start the backend server:
                    ```bash
                    cd backend
                    uvicorn main:app --reload --port 8000
                    ```
                    """)
                except Exception as e:
                    st.error(f"Unexpected error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_viz:
    st.markdown("## 📊 Model Visualizations")
    st.info("Run `notebooks/train_models.py` first to generate these visualizations.")

    BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR   = os.path.join(os.path.dirname(BACKEND_DIR), "models")

    def show_image(fname, caption):
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            st.image(path, caption=caption, use_column_width=True)
        else:
            st.warning(f"Image not found: {fname}. Train models first.")

    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "🩺 Diabetes", "❤️ Heart Disease", "🫀 Liver Disease", "📈 Comparison"
    ])

    with viz_tab1:
        col1, col2 = st.columns(2)
        with col1:
            show_image("diabetes_heatmap.png", "Diabetes – Correlation Heatmap")
        with col2:
            show_image("diabetes_evaluation.png", "Diabetes – Confusion Matrix & Feature Importance")
        show_image("diabetes_roc.png", "Diabetes – ROC Curve")

    with viz_tab2:
        col1, col2 = st.columns(2)
        with col1:
            show_image("heart_heatmap.png", "Heart Disease – Correlation Heatmap")
        with col2:
            show_image("heart_evaluation.png", "Heart Disease – Confusion Matrix & Feature Importance")
        show_image("heart_roc.png", "Heart Disease – ROC Curve")

    with viz_tab3:
        col1, col2 = st.columns(2)
        with col1:
            show_image("liver_heatmap.png", "Liver Disease – Correlation Heatmap")
        with col2:
            show_image("liver_evaluation.png", "Liver Disease – Confusion Matrix & Feature Importance")
        show_image("liver_roc.png", "Liver Disease – ROC Curve")

    with viz_tab4:
        show_image("roc_comparison.png", "ROC Curve Comparison – All Models")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    ## 📋 Project Documentation

    ### 🏗️ Project Architecture
    ```
    multi_disease_predictor/
    ├── data/                    # CSV datasets
    │   ├── diabetes.csv
    │   ├── heart_disease.csv
    │   └── liver_disease.csv
    ├── models/                  # Saved ML artifacts
    │   ├── *_model.pkl          # Trained XGBoost models
    │   ├── *_scaler.pkl         # StandardScaler objects
    │   ├── *_encoders.pkl       # LabelEncoder objects
    │   ├── *_features.pkl       # Feature name lists
    │   └── *.png                # Evaluation visualizations
    ├── backend/
    │   └── main.py              # FastAPI server
    ├── frontend/
    │   └── app.py               # Streamlit UI (this file)
    ├── notebooks/
    │   ├── train_models.ipynb   # Jupyter notebook
    │   └── train_models.py      # Standalone training script
    └── requirements.txt
    ```

    ### 🔬 Data Processing Pipeline
    | Step | Action |
    |------|--------|
    | 1 | Load CSV datasets |
    | 2 | Replace biologically impossible zeros with NaN |
    | 3 | Impute missing values with column median |
    | 4 | Encode categorical variables (LabelEncoder) |
    | 5 | Feature engineering (ratios, categories) |
    | 6 | Standardize features (StandardScaler) |
    | 7 | Train/test split (80/20, stratified) |
    | 8 | Train XGBoost classifier |
    | 9 | Evaluate & save model artifacts |

    ### 🤖 Model Details
    All three models use **XGBoost** with these parameters:
    - `n_estimators = 200`
    - `max_depth = 4`
    - `learning_rate = 0.1`
    - `subsample = 0.8`
    - `colsample_bytree = 0.8`

    ### 🎯 Risk Level Classification
    | Risk Level | Probability Range | Colour |
    |-----------|-------------------|--------|
    | 🟢 Low | < 35% | Green |
    | 🟡 Medium | 35% – 65% | Yellow |
    | 🔴 High | > 65% | Red |

    ### ▶️ Quick Start
    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Train models
    python notebooks/train_models.py

    # 3. Start backend (Terminal 1)
    cd backend && uvicorn main:app --reload --port 8000

    # 4. Start frontend (Terminal 2)
    cd frontend && streamlit run app.py
    ```
    """)
