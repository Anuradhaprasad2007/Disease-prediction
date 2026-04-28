"""
backend/main.py
───────────────
FastAPI backend for the Multi-Disease Prediction System.

Endpoints:
  GET  /           → health check
  POST /predict    → accepts patient data, returns predictions for all diseases
  GET  /diseases   → list supported diseases + required input fields

Run:
    cd backend
    uvicorn main:app --reload --port 8000
"""

import os
import sys
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ── Resolve model directory ───────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(os.path.dirname(BACKEND_DIR), "models")

# ── Load models, scalers, encoders ───────────────────────────────────────────
def load(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return None
    return joblib.load(path)

models_loaded = {
    "diabetes": {
        "model":    load("diabetes_model.pkl"),
        "scaler":   load("diabetes_scaler.pkl"),
        "features": load("diabetes_features.pkl"),
        "encoders": None,
    },
    "heart": {
        "model":    load("heart_model.pkl"),
        "scaler":   load("heart_scaler.pkl"),
        "features": load("heart_features.pkl"),
        "encoders": load("heart_encoders.pkl"),
    },
    "liver": {
        "model":    load("liver_model.pkl"),
        "scaler":   load("liver_scaler.pkl"),
        "features": load("liver_features.pkl"),
        "encoders": load("liver_encoders.pkl"),
    },
}

MODELS_AVAILABLE = all(
    v["model"] is not None for v in models_loaded.values()
)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multi-Disease Predictor API",
    description="XGBoost-powered API for predicting Diabetes, Heart Disease, and Liver Disease",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow Streamlit frontend
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic input schemas ────────────────────────────────────────────────────

class DiabetesInput(BaseModel):
    Pregnancies: float = Field(..., ge=0, le=20,  description="Number of pregnancies")
    Glucose: float     = Field(..., ge=0, le=300,  description="Plasma glucose (mg/dL)")
    BloodPressure: float = Field(..., ge=0, le=180, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm)")
    Insulin: float     = Field(..., ge=0, le=900,  description="2-Hour serum insulin (mu U/ml)")
    BMI: float         = Field(..., ge=0, le=80,   description="Body mass index")
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3, description="Diabetes pedigree function")
    Age: float         = Field(..., ge=1, le=120,  description="Age (years)")


class HeartInput(BaseModel):
    age: float       = Field(..., ge=1, le=120)
    sex: str         = Field(..., description="male or female")
    cp: str          = Field(..., description="chest pain type: typical angina | atypical angina | non-anginal | asymptomatic")
    trestbps: float  = Field(..., ge=80, le=250, description="Resting blood pressure")
    chol: float      = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int         = Field(..., ge=0, le=1,    description="Fasting blood sugar > 120 mg/dl (1=True)")
    restecg: int     = Field(..., ge=0, le=2,    description="Resting ECG results (0/1/2)")
    thalach: float   = Field(..., ge=60, le=250, description="Max heart rate achieved")
    exang: int       = Field(..., ge=0, le=1,    description="Exercise induced angina (1=Yes)")
    oldpeak: float   = Field(..., ge=0, le=10,   description="ST depression induced by exercise")
    slope: int       = Field(..., ge=0, le=2,    description="Slope of peak exercise ST segment")
    ca: int          = Field(..., ge=0, le=4,    description="Number of major vessels coloured by fluoroscopy")
    thal: str        = Field(..., description="Thalassemia: normal | fixed defect | reversable defect")


class LiverInput(BaseModel):
    Age: float    = Field(..., ge=1, le=120)
    Gender: str   = Field(..., description="Male or Female")
    Total_Bilirubin: float             = Field(..., ge=0)
    Direct_Bilirubin: float            = Field(..., ge=0)
    Alkaline_Phosphotase: float        = Field(..., ge=0)
    Alamine_Aminotransferase: float    = Field(..., ge=0)
    Aspartate_Aminotransferase: float  = Field(..., ge=0)
    Total_Protiens: float              = Field(..., ge=0)
    Albumin: float                     = Field(..., ge=0)
    Albumin_and_Globulin_Ratio: float  = Field(..., ge=0)


class PredictAllInput(BaseModel):
    diabetes: Optional[DiabetesInput] = None
    heart:    Optional[HeartInput]    = None
    liver:    Optional[LiverInput]    = None


# ── Risk level helper ─────────────────────────────────────────────────────────
def risk_level(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Medium"
    else:
        return "High"


# ── Feature preparation helpers ───────────────────────────────────────────────

def prepare_diabetes(data: DiabetesInput) -> list:
    d = data.dict()
    # Feature engineering (must match training)
    bmi = d["BMI"]
    if bmi < 18.5:      bmi_cat = 0
    elif bmi < 25:      bmi_cat = 1
    elif bmi < 30:      bmi_cat = 2
    else:               bmi_cat = 3

    age = d["Age"]
    if age < 30:        age_grp = 0
    elif age < 45:      age_grp = 1
    elif age < 60:      age_grp = 2
    else:               age_grp = 3

    gi_ratio = d["Glucose"] / (d["Insulin"] + 1)

    return [
        d["Pregnancies"], d["Glucose"], d["BloodPressure"], d["SkinThickness"],
        d["Insulin"], d["BMI"], d["DiabetesPedigreeFunction"], d["Age"],
        bmi_cat, age_grp, gi_ratio,
    ]


def prepare_heart(data: HeartInput, encoders: dict) -> list:
    d = data.dict()

    # Encode categorical
    sex_enc  = encoders["sex"].transform([d["sex"]])[0]
    cp_enc   = encoders["cp"].transform([d["cp"]])[0] if d["cp"] in encoders["cp"].classes_ else 0
    thal_val = d["thal"] if d["thal"] in encoders["thal"].classes_ else "normal"
    thal_enc = encoders["thal"].transform([thal_val])[0]

    age_thalach = d["age"] / (d["thalach"] + 1)
    bp_chol     = d["trestbps"] * d["chol"] / 10000

    return [
        d["age"], sex_enc, cp_enc, d["trestbps"], d["chol"], d["fbs"],
        d["restecg"], d["thalach"], d["exang"], d["oldpeak"], d["slope"],
        d["ca"], thal_enc, age_thalach, bp_chol,
    ]


def prepare_liver(data: LiverInput, encoders: dict) -> list:
    d = data.dict()
    gender_enc = encoders["Gender"].transform([d["Gender"]])[0]

    ast_alt     = d["Aspartate_Aminotransferase"] / (d["Alamine_Aminotransferase"] + 1)
    bili_ratio  = d["Direct_Bilirubin"] / (d["Total_Bilirubin"] + 1)
    prot_alb    = d["Total_Protiens"] - d["Albumin"]

    return [
        d["Age"], gender_enc, d["Total_Bilirubin"], d["Direct_Bilirubin"],
        d["Alkaline_Phosphotase"], d["Alamine_Aminotransferase"],
        d["Aspartate_Aminotransferase"], d["Total_Protiens"],
        d["Albumin"], d["Albumin_and_Globulin_Ratio"],
        ast_alt, bili_ratio, prot_alb,
    ]


def predict_one(disease: str, features: list) -> dict:
    """Run model inference for a single disease."""
    cfg = models_loaded[disease]
    if cfg["model"] is None:
        return {"error": f"{disease} model not loaded. Run train_models.py first."}

    X = np.array(features).reshape(1, -1)
    X_scaled = cfg["scaler"].transform(X)
    pred  = int(cfg["model"].predict(X_scaled)[0])
    prob  = float(cfg["model"].predict_proba(X_scaled)[0][1])

    return {
        "prediction":   pred,
        "result":       "Positive" if pred == 1 else "Negative",
        "probability":  round(prob * 100, 2),
        "risk_level":   risk_level(prob),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status": "running",
        "models_loaded": MODELS_AVAILABLE,
        "message": "Multi-Disease Predictor API is live!",
    }


@app.get("/diseases")
def list_diseases():
    return {
        "supported_diseases": ["diabetes", "heart", "liver"],
        "note": "Send data for one or more diseases in a single /predict call.",
    }


@app.post("/predict")
def predict(data: PredictAllInput):
    if not MODELS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Models not found. Please run notebooks/train_models.py first."
        )

    results = {}

    if data.diabetes:
        feats = prepare_diabetes(data.diabetes)
        results["diabetes"] = predict_one("diabetes", feats)

    if data.heart:
        encoders = models_loaded["heart"]["encoders"]
        feats = prepare_heart(data.heart, encoders)
        results["heart"] = predict_one("heart", feats)

    if data.liver:
        encoders = models_loaded["liver"]["encoders"]
        feats = prepare_liver(data.liver, encoders)
        results["liver"] = predict_one("liver", feats)

    if not results:
        raise HTTPException(status_code=400, detail="No disease data provided.")

    return {"predictions": results}


# ── Optional: individual endpoints ───────────────────────────────────────────

@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    if models_loaded["diabetes"]["model"] is None:
        raise HTTPException(status_code=503, detail="Diabetes model not loaded.")
    feats = prepare_diabetes(data)
    return predict_one("diabetes", feats)


@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    if models_loaded["heart"]["model"] is None:
        raise HTTPException(status_code=503, detail="Heart model not loaded.")
    encoders = models_loaded["heart"]["encoders"]
    feats = prepare_heart(data, encoders)
    return predict_one("heart", feats)


@app.post("/predict/liver")
def predict_liver(data: LiverInput):
    if models_loaded["liver"]["model"] is None:
        raise HTTPException(status_code=503, detail="Liver model not loaded.")
    encoders = models_loaded["liver"]["encoders"]
    feats = prepare_liver(data, encoders)
    return predict_one("liver", feats)
