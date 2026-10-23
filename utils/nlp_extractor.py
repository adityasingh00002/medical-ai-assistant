"""
nlp_extractor.py
----------------
This is the "NLP" component of the pipeline. It converts free-text user
input like:

    "I have headache and fever since 2 days, feeling very tired"

into a structured binary symptom vector matching the columns the
Disease Prediction Model was trained on.

Approach (deliberately dependency-light so it runs anywhere, but this
is exactly the kind of module you can extend with spaCy /
scispaCy / a medical NER model for a stronger project write-up):

1. Text normalization (lowercase, strip punctuation)
2. Synonym / phrase mapping ("running nose" -> "runny_nose",
   "throwing up" -> "vomiting", etc.) via a curated medical synonym
   dictionary
2. Fuzzy matching (difflib) to catch typos/variants that aren't in the
   synonym dictionary ("hedache" -> "headache")
3. Negation handling ("no fever", "not coughing") so negated symptoms
   are NOT counted as present -- an important correctness detail to
   mention in your viva.

In your report, you can describe this as a rule-based + fuzzy-matching
NLP pipeline, and note that it could be upgraded to a transformer-based
medical NER model (e.g. BioBERT / a fine-tuned spaCy model) as future
scope.
"""

import re
import difflib

# Canonical symptom -> list of natural-language phrases/synonyms
SYMPTOM_SYNONYMS = {
    "fever": ["fever", "high temperature", "temperature", "feverish"],
    "headache": ["headache", "head ache", "head pain", "migraine pain"],
    "cough": ["cough", "coughing"],
    "sore_throat": ["sore throat", "throat pain", "scratchy throat"],
    "runny_nose": ["runny nose", "running nose", "nose running"],
    "body_ache": ["body ache", "body pain", "muscle pain", "body aches", "aching body"],
    "fatigue": ["fatigue", "tired", "tiredness", "exhausted", "weakness", "low energy"],
    "nausea": ["nausea", "nauseous", "feeling sick", "queasy"],
    "vomiting": ["vomiting", "throwing up", "vomit", "puking"],
    "diarrhea": ["diarrhea", "loose motion", "loose motions", "watery stool"],
    "abdominal_pain": ["abdominal pain", "stomach pain", "stomach ache", "belly pain"],
    "chest_pain": ["chest pain", "pain in chest"],
    "shortness_of_breath": ["shortness of breath", "breathlessness", "difficulty breathing", "cant breathe"],
    "dizziness": ["dizziness", "dizzy", "lightheaded", "light headed"],
    "rash": ["rash", "skin rash", "red spots"],
    "joint_pain": ["joint pain", "pain in joints", "joints paining"],
    "chills": ["chills", "shivering", "feeling cold"],
    "loss_of_appetite": ["loss of appetite", "not feeling hungry", "no appetite"],
    "loss_of_smell": ["loss of smell", "cant smell", "no smell"],
    "sneezing": ["sneezing", "sneeze"],
    "burning_urination": ["burning urination", "burning while urinating", "burning pee"],
    "frequent_urination": ["frequent urination", "peeing a lot", "urinating frequently"],
    "back_pain": ["back pain", "pain in back"],
    "constipation": ["constipation", "constipated"],
    "itching": ["itching", "itchy", "itchy skin"],
    "swelling": ["swelling", "swollen"],
    "weight_loss": ["weight loss", "losing weight", "lost weight"],
    "excessive_thirst": ["excessive thirst", "very thirsty", "always thirsty"],
    "blurred_vision": ["blurred vision", "blurry vision", "vision problem"],
    "night_sweats": ["night sweats", "sweating at night"],
}

NEGATION_WORDS = ["no", "not", "without", "never", "n't", "dont", "don't"]


def _normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_negated(text, phrase_start_idx):
    """Check the ~4 words immediately before the matched phrase for a
    negation word, e.g. 'no fever', 'not coughing'."""
    preceding = text[:phrase_start_idx].split()[-4:]
    return any(neg in preceding for neg in NEGATION_WORDS)


def extract_symptoms(user_text, symptom_columns, fuzzy_cutoff=0.82):
    """
    Parameters
    ----------
    user_text : str
        Raw free-text symptom description from the user.
    symptom_columns : list[str]
        The ordered list of canonical symptom names the ML model expects
        (loaded from models/symptom_list.pkl).
    fuzzy_cutoff : float
        difflib similarity threshold for fuzzy word matching fallback.

    Returns
    -------
    dict:
        {
          "vector": {symptom: 0/1, ...}   # matches symptom_columns order
          "matched_symptoms": [...],       # human-readable list found
          "negated_symptoms": [...],       # symptoms explicitly denied
          "unrecognized_tokens": [...]     # words we couldn't map (useful
                                            # to show the user / log for
                                            # dataset improvement)
        }
    """
    norm_text = _normalize(user_text)
    found = {}
    negated = []

    # 1. Synonym/phrase matching
    for canonical, phrases in SYMPTOM_SYNONYMS.items():
        for phrase in sorted(phrases, key=len, reverse=True):
            idx = norm_text.find(phrase)
            if idx != -1:
                if _is_negated(norm_text, idx):
                    negated.append(canonical)
                else:
                    found[canonical] = True
                break  # stop at first matching phrase for this symptom

    # 2. Fuzzy fallback for single-word symptoms not already found,
    #    to catch typos e.g. "hedache", "vomitting"
    words = norm_text.split()
    single_word_map = {
        k.replace("_", ""): k for k in SYMPTOM_SYNONYMS if "_" not in k
    }
    unrecognized = []
    for w in words:
        if w in NEGATION_WORDS or len(w) < 4:
            continue
        already_matched = any(w in phrase for phrase in
                               [p for phrases in SYMPTOM_SYNONYMS.values() for p in phrases])
        if already_matched:
            continue
        close = difflib.get_close_matches(
            w, single_word_map.keys(), n=1, cutoff=fuzzy_cutoff
        )
        if close:
            canonical = single_word_map[close[0]]
            if canonical not in found:
                found[canonical] = True
        else:
            unrecognized.append(w)

    # Build the ordered binary vector expected by the ML model
    vector = {sym: (1 if sym in found else 0) for sym in symptom_columns}

    return {
        "vector": vector,
        "matched_symptoms": list(found.keys()),
        "negated_symptoms": negated,
        "unrecognized_tokens": unrecognized,
    }


if __name__ == "__main__":
    # quick manual test
    demo_symptoms = list(SYMPTOM_SYNONYMS.keys())
    test_text = "I have headache and fever since 2 days, no vomiting but feeling very tired"
    result = extract_symptoms(test_text, demo_symptoms)
    print("Input:", test_text)
    print("Matched:", result["matched_symptoms"])
    print("Negated:", result["negated_symptoms"])
    print("Unrecognized tokens:", result["unrecognized_tokens"])
