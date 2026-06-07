# api/main.py
# API FastAPI pour SenSante - Assistant pre-diagnostic medical
import os
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Charger les variables d'environnement
load_dotenv()

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
# Charger les variables d'environnement
load_dotenv()

# Client Groq (chargé au démarrage)
groq_client = None

groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialisé.")
else:
    print(
        "ATTENTION : GROQ_API_KEY non trouvée. "
        "/explain sera désactivé."
    )


class ExplainInput(BaseModel):
    diagnostic: str = Field(
        ...,
        description="Diagnostic prédit par le modèle"
    )

    probabilite: float = Field(
        ...,
        description="Probabilité du diagnostic"
    )

    age: int = Field(...)

    sexe: str = Field(...)

    temperature: float = Field(...)

    region: str = Field(...)


class ExplainOutput(BaseModel):
    explication: str = Field(
        ...,
        description="Explication en français"
    )

    modele_llm: str = Field(
        default="llama-3.1-8b-instant",
        description="Modèle LLM utilisé"
    )

    
SYSTEM_PROMPT = """
Tu es un assistant médical sénégalais.

Tu reçois un diagnostic et des données patient.
Explique le résultat en français simple,
comme un médecin parlerait à son patient.

Sois rassurant mais recommande toujours
une consultation médicale.

Maximum 3 phrases.

Ne fais JAMAIS de diagnostic toi-même.
Tu expliques uniquement le diagnostic fourni.
"""


@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """
    Expliquer un diagnostic en français avec un LLM.
    """

    # Vérifier si le client Groq est disponible
    if not groq_client:
        return ExplainOutput(
            explication=(
                "Service d'explication indisponible. "
                "Clé API non configurée."
            ),
            modele_llm="aucun"
        )

    # Construire le prompt utilisateur
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, "
        f"région {data.region}\n"
        f"Température : {data.temperature} °C\n"
        f"Diagnostic du modèle : {data.diagnostic} "
        f"(probabilité {data.probabilite:.0%})\n"
        f"Explique ce résultat au patient."
    )

    try:
        # Appel au modèle Groq
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens=200,
            temperature=0.3
        )

        explication = response.choices[0].message.content

    except Exception as e:
        explication = (
            f"Erreur lors de l'appel au LLM : {str(e)}"
        )

    return ExplainOutput(
        explication=explication
    )


