# api/main.py
# API FastAPI pour SenSante - Assistant pre-diagnostic medical

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Creer l'application
app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.2.0"
)

# Route de base : verifier que l'API fonctionne
@app.get("/health")
def health_check():
    """Verification de l'etat de l'API."""
    
    return {
        "status": "ok",
        "message": "SenSante API is running"
    }


# --- Schemas Pydantic ---

class PatientInput(BaseModel):
    """Donnees d'entree : les symptomes d'un patient."""

    age: int = Field(
        ...,
        ge=0,
        le=120,
        description="Age en annees"
    )

    sexe: str = Field(
        ...,
        description="Sexe : M ou F"
    )

    temperature: float = Field(
        ...,
        ge=35.0,
        le=42.0,
        description="Temperature en Celsius"
    )

    tension_sys: int = Field(
        ...,
        ge=60,
        le=250,
        description="Tension systolique"
    )

    toux: bool = Field(
        ...,
        description="Presence de toux"
    )

    fatigue: bool = Field(
        ...,
        description="Presence de fatigue"
    )

    maux_tete: bool = Field(
        ...,
        description="Presence de maux de tete"
    )

    region: str = Field(
        ...,
        description="Region du Senegal"
    )


class DiagnosticOutput(BaseModel):
    """Donnees de sortie : le resultat du diagnostic."""

    diagnostic: str = Field(
        ...,
        description="Diagnostic predit"
    )
    probabilite: float = Field(
        ...,
        description="Probabilite du diagnostic"
    )

    confiance: str = Field(
        ...,
        description="Niveau de confiance"
    )

    message: str = Field(
        ...,
        description="Recommandation"
    )


# --- Charger le modele et les encodeurs au demarrage ---

print("Chargement du modele...")
model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
print(f"Modele charge : {type(model).__name__}")
print(f"Classes : {list(model.classes_)}")

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    """
    Prédire un diagnostic à partir des symptômes d'un patient.
    Reçoit les symptômes en JSON, renvoie le diagnostic,
    la probabilité et une recommandation.
    """

    # 1. Encoder les variables catégoriques
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}. Utiliser M ou F."
        )

    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Région inconnue : {patient.region}"
        )

    # 2. Construire le vecteur de features
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])

    # 3. Prédire
    diagnostic = model.predict(features)[0]
    probas = model.predict_proba(features)[0]
    proba_max = float(probas.max())

    # 4. Déterminer le niveau de confiance
    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    # 5. Générer la recommandation
    messages = {
        "palu": "Suspicion de paludisme. Consultez un médecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandés.",
        "typh": "Suspicion de typhoïde. Consultation médicale nécessaire.",
        "sain": "Pas de pathologie détectée. Continuez à surveiller."
    }

    # 6. Renvoyer le résultat
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(
            diagnostic,
            "Consultez un médecin."
        )
    )



@app.get("/model-info")
def model_info():
    return {
        "type_modele": type(model).__name__,
        "nombre_arbres": model.n_estimators,
        "classes_possibles": model.classes_.tolist(),
        "nombre_features": model.n_features_in_
    }



# Autoriser les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,

    # En développement : tout accepter
    allow_origins=["*"],

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)