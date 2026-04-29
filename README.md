# 🏥 MediPredict – Multi-Disease Prediction System

An end-to-end Machine Learning project that predicts **Diabetes**, **Heart Disease**, and **Liver Disease** using XGBoost models, with a FastAPI backend and Streamlit frontend.

---

## 📁 Project Structure

```
multi_disease_predictor/
├── data/
│   ├── diabetes.csv              # Pima Indians Diabetes dataset (slightly unclean)
│   ├── heart_disease.csv         # Cleveland Heart Disease dataset (with categorical cols)
│   └── liver_disease.csv         # Indian Liver Patient dataset
│
├── models/                       # Auto-generated after training
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── diabetes_features.pkl
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── heart_features.pkl
│   ├── heart_encoders.pkl
│   ├── liver_model.pkl
│   ├── liver_scaler.pkl
│   ├── liver_features.pkl
│   ├── liver_encoders.pkl
│   └── *.png                     # Evaluation charts
│
├── backend/
│   └── main.py                   # FastAPI REST API
│
├── frontend/
│   └── app.py                    # Streamlit web interface
│
├── notebooks/
│   ├── train_models.ipynb        # Step-by-step Jupyter notebook
│   └── train_models.py           # Standalone training script
│
└── requirements.txt
```

---

## ⚡ Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Models
```bash
python notebooks/train_models.py
```
This will:
- Clean & preprocess all 3 datasets
- Engineer new features
- Train XGBoost models
- Save models to `models/`
- Generate evaluation charts (confusion matrices, ROC curves, feature importance)

### Step 3: Start the Backend API
```bash

python -m uvicorn main:app --reload
```
> API will be live at: http://localhost:8000
> Docs at: http://localhost:8000/docs

### Step 4: Start the Streamlit Frontend
Open a **new terminal**:
```bash

python -m streamlit run app.py
```
> App will open at: http://localhost:8501

---

## 🔬 Data Processing Steps

### Diabetes Dataset
| Issue | Fix |
|-------|-----|
| Zeros in Glucose, BP, BMI, etc. | Replace with NaN → impute with median |
| No categorical columns | — |
| New features | BMI_Category, Age_Group, Glucose_Insulin_Ratio |

### Heart Disease Dataset
| Issue | Fix |
|-------|-----|
| `sex` column: "male"/"female" | LabelEncoder |
| `cp` column: text chest pain types | LabelEncoder |
| `thal` column: text thalassemia types | LabelEncoder |
| Missing values | Median imputation |
| New features | age_thalach_ratio, bp_chol_index |

### Liver Disease Dataset
| Issue | Fix |
|-------|-----|
| `Gender`: "Male"/"Female" | LabelEncoder |
| Target: 1=Disease, 2=No Disease | Converted to 0/1 |
| Class imbalance | `scale_pos_weight` in XGBoost |
| New features | AST_ALT_Ratio, Bilirubin_Ratio, Protein_Albumin_Diff |

---

## 🤖 Model Configuration

All diseases use **XGBoost** with:
```python
XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
```

---

## 📡 API Reference

### `GET /`
Health check + model status

### `GET /diseases`
List supported diseases

### `POST /predict`
Predict one or more diseases simultaneously.

**Request Body:**
```json
{
  "diabetes": {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 72,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 28.0,
    "DiabetesPedigreeFunction": 0.47,
    "Age": 33
  },
  "heart": {
    "age": 52,
    "sex": "male",
    "cp": "atypical angina",
    "trestbps": 130,
    "chol": 230,
    "fbs": 0,
    "restecg": 1,
    "thalach": 155,
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 0,
    "thal": "normal"
  }
}
```

**Response:**
```json
{
  "predictions": {
    "diabetes": {
      "prediction": 1,
      "result": "Positive",
      "probability": 72.4,
      "risk_level": "High"
    },
    "heart": {
      "prediction": 0,
      "result": "Negative",
      "probability": 23.1,
      "risk_level": "Low"
    }
  }
}
```

### `POST /predict/diabetes`
### `POST /predict/heart`
### `POST /predict/liver`
Individual prediction endpoints (same schema as above).

---

## 🎯 Risk Level Classification

| Level | Probability | Color |
|-------|------------|-------|
| 🟢 **Low** | < 35% | Green |
| 🟡 **Medium** | 35% – 65% | Yellow |
| 🔴 **High** | > 65% | Red |

---

## 📊 Evaluation Metrics

Each model is evaluated on:
- **Accuracy** – Overall correctness
- **Precision** – Of predicted positives, how many are truly positive
- **Recall** – Of actual positives, how many did we catch
- **F1 Score** – Harmonic mean of precision and recall
- **ROC AUC** – Area under the ROC curve
- **Confusion Matrix** – Visual breakdown of TP/TN/FP/FN

---

## 📦 Datasets

The project includes sample datasets. For better performance, replace with:
- **Diabetes**: [Pima Indians Diabetes (Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) – 768 rows
- **Heart Disease**: [Cleveland Heart Disease (UCI)](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) – 303 rows
- **Liver Disease**: [Indian Liver Patient Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/ILPD) – 583 rows

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. The predictions should not be used as a substitute for professional medical diagnosis. Always consult a licensed healthcare provider.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Model Storage | Joblib |
