"""Offline medical-information guidance. No LLM, API key, or paid service required.
This is educational information, not diagnosis or treatment advice.
"""


def generate_advice(disease, matched_symptoms, confidence=0.0):
    symptoms = ", ".join(matched_symptoms) if matched_symptoms else "the symptoms described"
    return {
        "explanation": (
            f"The symptom-based model matched {symptoms} and its dataset ranked {disease} highest. "
            "Many conditions can cause similar symptoms, so this result cannot confirm a condition."
        ),
        "home_remedies": [
            "Rest and drink fluids if you can do so safely.",
            "Keep a note of your symptoms, when they started, and whether they are getting worse.",
            "Choose light, nourishing food as tolerated.",
            "Do not start antibiotics or stop prescribed medicines based on this result."
        ],
        "see_doctor_if": [
            "Symptoms are severe, worsening, recurring, or not improving.",
            "You have persistent fever, signs of dehydration, or significant pain.",
            "You are pregnant, immunocompromised, have a chronic condition, or are concerned about a child or older adult."
        ],
        "disclaimer": (
            "Educational decision-support only: this model can be wrong and is not a diagnosis. "
            "For chest pain, severe breathing difficulty, fainting, confusion, or another emergency, seek urgent medical care."
        ),
        "advice_source": "offline_rule_based"
    }
