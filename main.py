from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib


# ==========================================
# 1. INITIALIZE FASTAPI
# ==========================================

app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="Predict 30-day hospital readmission risk",
    version="1.0"
)


# ==========================================
# 2. LOAD TRAINED ML MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "readmission_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]

FEATURES = bundle["features"]

config = bundle["preprocessing_config"]

print("ML Model Loaded Successfully!")
print("Model:", bundle["model_name"])
print("Features:", len(FEATURES))


# ==========================================
# 3. PATIENT INPUT DATA
# ==========================================

class PatientData(BaseModel):

    encounter_id: int

    age: str
    gender: str

    time_in_hospital: int = 1

    num_lab_procedures: int = 0
    num_procedures: int = 0
    num_medications: int = 0

    number_outpatient: int = 0
    number_emergency: int = 0
    number_inpatient: int = 0

    number_diagnoses: int = 1

    diag_1: str = "Missing"
    diag_2: str = "Missing"
    diag_3: str = "Missing"

    admission_type_id: int = 1
    discharge_disposition_id: int = 1
    admission_source_id: int = 1

    medical_specialty: str = "Unknown"

    payer_code: str = "Unknown"

    max_glu_serum: str = "Not tested"
    A1Cresult: str = "Not tested"

    change: str = "No"
    diabetesMed: str = "No"

    # Medication information

    metformin: str = "No"
    repaglinide: str = "No"
    nateglinide: str = "No"
    chlorpropamide: str = "No"
    glimepiride: str = "No"
    acetohexamide: str = "No"
    glipizide: str = "No"
    glyburide: str = "No"
    tolbutamide: str = "No"
    pioglitazone: str = "No"
    rosiglitazone: str = "No"
    acarbose: str = "No"
    miglitol: str = "No"
    troglitazone: str = "No"
    tolazamide: str = "No"
    insulin: str = "No"
    glyburide_metformin: str = "No"
    glipizide_metformin: str = "No"
    glimepiride_pioglitazone: str = "No"
    metformin_rosiglitazone: str = "No"
    metformin_pioglitazone: str = "No"


# ==========================================
# 4. MISSING VALUE HANDLING
# ==========================================

def fill_missing(df):

    df = df.copy()

    for col in [
        "race",
        "payer_code",
        "medical_specialty"
    ]:

        if col not in df.columns:
            df[col] = "Unknown"

        df[col] = df[col].fillna("Unknown")

    for col in ["diag_1", "diag_2", "diag_3"]:

        df[col] = df[col].fillna("Missing")

    for col in ["max_glu_serum", "A1Cresult"]:

        df[col] = (
            df[col]
            .fillna("Not tested")
            .replace("None", "Not tested")
        )

    return df


# ==========================================
# 5. DIAGNOSIS GROUPING
# ==========================================

def diag_to_group(code):

    code = str(code)

    if code == "Missing":
        return "Missing"

    if code.startswith(("V", "E")):
        return "Other"

    try:

        value = int(float(code))

    except (ValueError, TypeError):

        return "Other"

    if value == 250:
        return "Diabetes"

    if 390 <= value <= 459 or value == 785:
        return "Circulatory"

    if 460 <= value <= 519 or value == 786:
        return "Respiratory"

    if 520 <= value <= 579 or value == 787:
        return "Digestive"

    if 800 <= value <= 999:
        return "Injury"

    if 710 <= value <= 739:
        return "Musculoskeletal"

    if 580 <= value <= 629 or value == 788:
        return "Genitourinary"

    if 140 <= value <= 239:
        return "Neoplasms"

    return "Other"


# ==========================================
# 6. FEATURE ENGINEERING
# ==========================================

def engineer_features(df):

    df = df.copy()

    # Age conversion

    age_midpoint = config["age_midpoint"]

    df["age_num"] = df["age"].map(age_midpoint)

    if df["age_num"].isna().any():

        raise ValueError("Invalid age bracket")

    df["age_group"] = pd.cut(
        df["age_num"],
        bins=[0, 30, 50, 70, 80, 100],
        labels=[
            "<30",
            "30-49",
            "50-69",
            "70-79",
            "80+"
        ]
    ).astype(str)

    # Previous hospital visits

    visit_cols = [
        "number_outpatient",
        "number_emergency",
        "number_inpatient"
    ]

    df["total_prior_visits"] = (
        df[visit_cols].sum(axis=1)
    )

    df["prior_inpatient_bucket"] = (
        df["number_inpatient"]
        .clip(upper=2)
        .map({
            0: "0",
            1: "1",
            2: "2+"
        })
    )

    # Diagnosis groups

    for col in ["diag_1", "diag_2", "diag_3"]:

        df[col + "_group"] = (
            df[col].apply(diag_to_group)
        )

    # Admission type

    df["admission_type"] = (
        df["admission_type_id"]
        .map(config["admission_type_map"])
        .fillna("Unknown")
    )

    # Discharge group

    df["discharge_group"] = (
        df["discharge_disposition_id"]
        .map(config["discharge_group_map"])
        .fillna("Unknown")
    )

    # Admission source

    df["admission_source_group"] = (
        df["admission_source_id"]
        .map(config["source_group_map"])
        .fillna("Unknown")
    )

    # Medical specialty

    top_specialties = config["top_specialties"]

    df["medical_specialty_grp"] = (
        df["medical_specialty"].where(
            df["medical_specialty"].isin(
                top_specialties + ["Unknown"]
            ),
            "Other"
        )
    )

    # Payer code

    top_payers = config["top_payers"]

    df["payer_code_grp"] = (
        df["payer_code"].where(
            df["payer_code"].isin(
                top_payers + ["Unknown"]
            ),
            "Other"
        )
    )

    # Medication features

    med_columns = config["med_columns"]

    for col in med_columns:

        if col not in df.columns:
            df[col] = "No"

    df["n_meds_active"] = (
        df[med_columns] != "No"
    ).sum(axis=1)

    df["n_meds_changed"] = (
        df[med_columns]
        .isin(["Up", "Down"])
        .sum(axis=1)
    )

    return df


# ==========================================
# 7. PREDICTION FUNCTION
# ==========================================

def predict_risk(patient):

    df = pd.DataFrame([patient])

    # Missing value handling

    df = fill_missing(df)

    # Feature engineering

    df = engineer_features(df)

    # Select training features

    X = df[FEATURES]

    # Predict probability

    probability = model.predict_proba(X)[0][1]

    # Threshold

    threshold = bundle["threshold"]

    cuts = bundle["risk_band_cutoffs"]

    # Risk classification

    if probability >= cuts["high_from"]:

        risk = "High"

    elif probability >= cuts["low_below"]:

        risk = "Medium"

    else:

        risk = "Low"

    return {

        "encounter_id": patient["encounter_id"],

        "readmission_probability": round(
            float(probability), 4
        ),

        "risk_percentage": round(
            float(probability * 100), 2
        ),

        "risk_level": risk,

        "flagged_high_risk": bool(
            probability >= threshold
        )
    }


# ==========================================
# 8. HOME API
# ==========================================

@app.get("/")

def home():

    return {

        "message": "Hospital Readmission Prediction API",

        "status": "Running"

    }


# ==========================================
# 9. MODEL INFORMATION
# ==========================================

@app.get("/model-info")

def model_info():

    return {

        "model_name": bundle["model_name"],

        "features_count": len(FEATURES),

        "threshold": bundle["threshold"],

        "base_rate": bundle["base_rate"],

        "sklearn_version": bundle["sklearn_version"]

    }


# ==========================================
# 10. PREDICTION API
# ==========================================

@app.post("/predict")

def predict(patient: PatientData):

    try:

        patient_data = patient.model_dump()

        result = predict_risk(patient_data)

        return result

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )