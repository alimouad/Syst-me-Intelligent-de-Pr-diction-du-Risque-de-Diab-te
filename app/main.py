import os
from functools import lru_cache

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "GlucoRiskAI")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "production")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


class PatientInput(BaseModel):
    pregnancies: int = Field(ge=0, le=20)
    glucose: float = Field(gt=0, le=400)
    blood_pressure: float = Field(gt=0, le=250)
    skin_thickness: float = Field(ge=0, le=150)
    insulin: float = Field(ge=0, le=1000)
    bmi: float = Field(gt=0, le=100)
    diabetes_pedigree_function: float = Field(ge=0, le=3)
    age: int = Field(ge=1, le=120)


@lru_cache(maxsize=1)
def load_production_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )


def patient_to_frame(patient: PatientInput) -> pd.DataFrame:
    return pd.DataFrame(
        [[
            patient.pregnancies,
            patient.glucose,
            patient.blood_pressure,
            patient.skin_thickness,
            patient.insulin,
            patient.bmi,
            patient.diabetes_pedigree_function,
            patient.age,
        ]],
        columns=FEATURE_NAMES,
    )

app = FastAPI(
    title="Diabetes Risk API",
    description="API for diabetes risk prediction",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Mouad is dying to make this API work, please be patient and wait for the next update."
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.post("/predict")
def predict(patient: PatientInput):
    try:
        model = load_production_model()
        prediction = str(model.predict(patient_to_frame(patient))[0])
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Production model is unavailable. "
                f"Check MLflow Registry: {error}"
            ),
        ) from error

    high_risk = prediction == "High Risk"

    return {
        "risk_category": prediction,
        "risk_level": "high" if high_risk else "low",
        "badge": "red" if high_risk else "green",
        "advice": (
            "Consultez rapidement un professionnel de sante et "
            "surveillez votre glycemie."
            if high_risk
            else "Maintenez une alimentation equilibree, une activite "
            "reguliere et un suivi medical."
        ),
        "model": f"{MODEL_NAME}@{MODEL_ALIAS}",
    }
