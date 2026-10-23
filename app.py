"""
app.py
------
Main Flask application for the Medical AI Assistant.

Pipeline per request (POST /predict):
  1. User free-text symptoms come in
  2. NLP module (utils/nlp_extractor.py) extracts a structured symptom vector
  3. ML model (models/disease_model.pkl) predicts the disease + confidence
  4. Specialist recommender (utils/specialist_recommender.py) maps disease
     -> specialist type + urgency
  5. Offline advisor provides general guidance without external API calls
  6. Consultation history is saved to MongoDB if it is configured

Run:
    python app.py
Then open http://localhost:5000
"""

import os
import pickle

from dotenv import load_dotenv
load_dotenv()  # Optional .env file; no API keys are required.

from flask import Flask, render_template, request, jsonify

from utils.nlp_extractor import extract_symptoms
from utils.specialist_recommender import get_specialist_info, find_nearby_specialists
from utils.llm_advisor import generate_advice
from utils.db import save_consultation, get_recent_consultations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model.pkl")
SYMPTOMS_PATH = os.path.join(BASE_DIR, "models", "symptom_list.pkl")

app = Flask(__name__)

# ---------------------------------------------------------------------
# Load ML artifacts once at startup (not per-request -> much faster)
# ---------------------------------------------------------------------
_model = None
_symptom_columns = None


def load_model():
    global _model, _symptom_columns
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model not found. Run `python data/build_dataset.py` then "
                "`python models/train_model.py` first."
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(SYMPTOMS_PATH, "rb") as f:
            _symptom_columns = pickle.load(f)
    return _model, _symptom_columns


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "prediction": "local-ml-model", "advice": "offline", "external_api_keys_required": False})


@app.route("/history")
def history():
    records = get_recent_consultations(20)
    return render_template("history.html", records=records)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    user_text = (data.get("symptoms") or "").strip()
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if not user_text:
        return jsonify({"error": "Please describe your symptoms."}), 400

    try:
        model, symptom_columns = load_model()
    except Exception as exc:
        app.logger.exception("Unable to load prediction model")
        return jsonify({"error": "The prediction model could not be loaded. Check that models/disease_model.pkl and models/symptom_list.pkl are present."}), 500

    # 1. NLP: text -> structured symptom vector
    nlp_result = extract_symptoms(user_text, symptom_columns)
    vector = nlp_result["vector"]

    if not any(vector.values()):
        return jsonify({
            "error": (
                "I couldn't recognize any known symptoms in that description. "
                "Try phrasing like: 'I have headache and fever'."
            ),
            "unrecognized_tokens": nlp_result["unrecognized_tokens"],
        }), 200

    # 2. ML: predict disease + confidence
    import pandas as pd
    X = pd.DataFrame([[vector[s] for s in symptom_columns]], columns=symptom_columns)
    proba = model.predict_proba(X)[0]
    classes = model.named_steps["clf"].classes_ if hasattr(model, "named_steps") else model.classes_
    best_idx = proba.argmax()
    predicted_disease = classes[best_idx]
    confidence = float(proba[best_idx])

    # top-3 alternatives, useful for showing "possible disease(s)" plural
    top3_idx = proba.argsort()[::-1][:3]
    top3 = [
        {"disease": classes[i], "confidence": round(float(proba[i]), 3)}
        for i in top3_idx
    ]

    # 3. Specialist recommendation
    specialist, urgency, urgency_note = get_specialist_info(predicted_disease)
    nearby = find_nearby_specialists(specialist, latitude, longitude)

    # 4. LLM: remedies / explanation / when-to-see-doctor
    advice = generate_advice(
        predicted_disease, nlp_result["matched_symptoms"], confidence
    )

    response = {
        "matched_symptoms": nlp_result["matched_symptoms"],
        "negated_symptoms": nlp_result["negated_symptoms"],
        "unrecognized_tokens": nlp_result["unrecognized_tokens"],
        "predicted_disease": predicted_disease,
        "confidence": round(confidence, 3),
        "top3_predictions": top3,
        "specialist": specialist,
        "urgency": urgency,
        "urgency_note": urgency_note,
        "nearby_specialists": nearby,
        "advice": advice,
    }

    # 5. Persist to MongoDB (no-op gracefully if DB isn't running)
    try:
        save_consultation({"user_input": user_text, **response})
    except Exception:
        app.logger.exception("Could not save consultation history; prediction will still be returned")

    return jsonify(response)


if __name__ == "__main__":
    # Start even if the model is missing so /api/health and the error page remain available.
    app.run(debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true", host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
