"""
build_dataset.py
-----------------
Generates a synthetic (but medically-reasonable) symptom -> disease dataset
for training the Disease Prediction Model.

In a real-world / research submission you would replace this with a public
dataset such as the "Disease Prediction Using Machine Learning" dataset on
Kaggle (132 symptoms x 41 diseases). The generation logic below is written
so that swapping in a real CSV (data/dataset.csv with the same column
format) requires zero code changes elsewhere in the project.
"""

import csv
import random

random.seed(42)

# Master symptom vocabulary (this becomes our feature columns / bag-of-symptoms)
SYMPTOMS = [
    "fever", "headache", "cough", "sore_throat", "runny_nose", "body_ache",
    "fatigue", "nausea", "vomiting", "diarrhea", "abdominal_pain",
    "chest_pain", "shortness_of_breath", "dizziness", "rash",
    "joint_pain", "chills", "loss_of_appetite", "loss_of_smell",
    "sneezing", "burning_urination", "frequent_urination", "back_pain",
    "constipation", "itching", "swelling", "weight_loss",
    "excessive_thirst", "blurred_vision", "night_sweats"
]

# Disease -> characteristic symptom sets (with weights so we can sample
# realistic partial-symptom presentations, mimicking real patients who
# rarely report every textbook symptom).
DISEASE_PROFILES = {
    "Common Cold": ["runny_nose", "sneezing", "sore_throat", "cough", "headache", "fatigue"],
    "Influenza (Flu)": ["fever", "body_ache", "chills", "headache", "fatigue", "cough", "sore_throat"],
    "COVID-19": ["fever", "cough", "loss_of_smell", "fatigue", "shortness_of_breath", "sore_throat", "body_ache"],
    "Migraine": ["headache", "dizziness", "nausea", "blurred_vision", "fatigue"],
    "Gastroenteritis": ["nausea", "vomiting", "diarrhea", "abdominal_pain", "fever", "loss_of_appetite"],
    "Typhoid": ["fever", "abdominal_pain", "headache", "loss_of_appetite", "fatigue", "constipation"],
    "Urinary Tract Infection": ["burning_urination", "frequent_urination", "abdominal_pain", "fever", "back_pain"],
    "Dengue Fever": ["fever", "rash", "joint_pain", "headache", "fatigue", "nausea"],
    "Allergic Rhinitis": ["sneezing", "runny_nose", "itching", "sore_throat"],
    "Bronchitis": ["cough", "chest_pain", "shortness_of_breath", "fatigue", "fever"],
    "Asthma": ["shortness_of_breath", "chest_pain", "cough", "fatigue"],
    "Type 2 Diabetes": ["excessive_thirst", "frequent_urination", "fatigue", "blurred_vision", "weight_loss"],
    "Hypertension": ["headache", "dizziness", "chest_pain", "blurred_vision"],
    "Chickenpox": ["rash", "itching", "fever", "fatigue", "loss_of_appetite"],
    "Tuberculosis": ["cough", "weight_loss", "night_sweats", "fever", "fatigue", "chest_pain"],
    "Food Poisoning": ["vomiting", "diarrhea", "abdominal_pain", "nausea", "fever"],
    "Anemia": ["fatigue", "dizziness", "blurred_vision", "headache"],
    "Arthritis": ["joint_pain", "swelling", "back_pain", "fatigue"],
    "Pneumonia": ["fever", "cough", "chest_pain", "shortness_of_breath", "fatigue", "chills"],
    "Malaria": ["fever", "chills", "headache", "body_ache", "nausea", "fatigue"],
}


def sample_row(disease, profile):
    """Create one training row: pick a realistic subset of the disease's
    symptoms (60-100% of them) + a small chance of 1-2 unrelated noise
    symptoms, mirroring how real patients self-report."""
    n_present = max(2, int(len(profile) * random.uniform(0.55, 1.0)))
    present = set(random.sample(profile, n_present))

    # small chance of 1 unrelated symptom (noise) to avoid the model
    # memorizing perfectly separable rules
    if random.random() < 0.15:
        noise_candidates = [s for s in SYMPTOMS if s not in profile]
        present.add(random.choice(noise_candidates))

    row = {s: (1 if s in present else 0) for s in SYMPTOMS}
    row["disease"] = disease
    return row


def main(n_per_disease=60, out_path="data/dataset.csv"):
    rows = []
    for disease, profile in DISEASE_PROFILES.items():
        for _ in range(n_per_disease):
            rows.append(sample_row(disease, profile))

    random.shuffle(rows)

    fieldnames = SYMPTOMS + ["disease"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows, {len(DISEASE_PROFILES)} diseases, "
          f"{len(SYMPTOMS)} symptom features -> {out_path}")


if __name__ == "__main__":
    main()
