"""Offline specialist recommendations with free map-search links (no API key)."""
from urllib.parse import quote_plus

DISEASE_SPECIALIST_MAP = {
    "Common Cold": ("General Physician", "routine", "Consider a clinician if symptoms worsen or persist."),
    "Influenza (Flu)": ("General Physician", "soon", "Seek medical advice for persistent fever or worsening symptoms."),
    "COVID-19": ("General Physician / Pulmonologist", "soon", "Seek urgent care for breathing difficulty."),
    "Migraine": ("Neurologist", "routine", "Consult a clinician for new, frequent, or severe headaches."),
    "Gastroenteritis": ("General Physician / Gastroenterologist", "soon", "Seek care for dehydration, blood in stool, or persistent vomiting."),
    "Typhoid": ("General Physician / Infectious Disease Specialist", "soon", "Prompt testing and clinician-directed treatment may be needed."),
    "Urinary Tract Infection": ("General Physician / Urologist", "soon", "Get assessed, especially with fever, back pain, or pregnancy."),
    "Dengue Fever": ("General Physician / Infectious Disease Specialist", "soon", "Seek prompt medical evaluation; avoid self-medicating."),
    "Allergic Rhinitis": ("General Physician / ENT Specialist", "routine", "Consult a clinician if symptoms persist or interfere with daily life."),
    "Bronchitis": ("General Physician / Pulmonologist", "soon", "Get assessed if breathing worsens or symptoms persist."),
    "Asthma": ("General Physician / Pulmonologist", "soon", "Severe breathing difficulty is an emergency."),
    "Type 2 Diabetes": ("General Physician / Endocrinologist", "soon", "A clinician can arrange appropriate testing."),
    "Hypertension": ("General Physician / Cardiologist", "soon", "Arrange a blood-pressure check; urgent symptoms need urgent care."),
    "Chickenpox": ("General Physician / Dermatologist", "routine", "Get medical advice, particularly for adults, pregnancy, or weakened immunity."),
    "Tuberculosis": ("General Physician / Pulmonologist", "soon", "Persistent cough or possible exposure should be assessed promptly."),
    "Food Poisoning": ("General Physician / Gastroenterologist", "soon", "Seek care for dehydration, blood in stool, or severe symptoms."),
    "Anemia": ("General Physician / Hematologist", "routine", "A clinician can assess symptoms and arrange blood tests."),
    "Arthritis": ("General Physician / Rheumatologist / Orthopedist", "routine", "Arrange assessment for persistent joint pain or swelling."),
    "Pneumonia": ("General Physician / Pulmonologist", "soon", "Prompt clinical evaluation is important, especially with breathing symptoms."),
    "Malaria": ("General Physician / Infectious Disease Specialist", "soon", "Fever after possible mosquito exposure warrants prompt testing."),
}
DEFAULT_SPECIALIST = ("General Physician", "soon", "Consult a qualified clinician for an assessment.")


def get_specialist_info(disease):
    return DISEASE_SPECIALIST_MAP.get(str(disease), DEFAULT_SPECIALIST)


def find_nearby_specialists(specialist_type, latitude=None, longitude=None):
    query = quote_plus(f"{specialist_type} near me")
    maps_url = f"https://www.google.com/maps/search/{query}"
    osm_url = f"https://www.openstreetmap.org/search?query={query}"
    if latitude is not None and longitude is not None:
        maps_url = f"https://www.google.com/maps/search/{query}/@{latitude},{longitude},13z"
    return {
        "available": False,
        "message": "No paid places API is used. Open a free map search to find nearby care; verify details and availability directly.",
        "results": [],
        "maps_url": maps_url,
        "openstreetmap_url": osm_url,
    }
