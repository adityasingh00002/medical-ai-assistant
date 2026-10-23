# Medical AI Assistant — free, API-key-free edition

A student project that connects a responsive HTML/CSS/JavaScript frontend to a Flask backend, a local scikit-learn symptom model, rule-based symptom extraction, specialist guidance, and optional consultation history.

> **Important:** This is an educational demo, not a medical device. The prediction model may be wrong and must not be used to diagnose, treat, or rule out a disease. For emergencies such as severe breathing difficulty, chest pain, fainting, or confusion, seek urgent medical care.

## What was changed

- Removed OpenAI, Gemini/LangChain, and Google Places API dependencies and calls.
- Advice is generated locally with cautious, rule-based text. It requires **no API key and no API billing**.
- Nearby-care links use Google Maps search and OpenStreetMap search URLs; the app does not call a paid places API or claim to verify listings.
- Frontend posts JSON to Flask `POST /predict`, handles errors, renders results safely, and supports mobile screens.
- Added `GET /api/health` to check that the backend is reachable.
- MongoDB is optional. Without it, predictions still work; history is not persisted.

## Project structure

```text
medical-ai-assistant/
├── app.py
├── requirements.txt
├── .env.example
├── data/
│   ├── build_dataset.py
│   └── dataset.csv
├── models/
│   ├── disease_model.pkl
│   ├── symptom_list.pkl
│   ├── label_classes.pkl
│   └── train_model.py
├── utils/
│   ├── nlp_extractor.py
│   ├── llm_advisor.py          # offline advice; no LLM API
│   ├── specialist_recommender.py
│   └── db.py                   # optional MongoDB history
├── templates/
│   ├── index.html
│   └── history.html
└── static/
    ├── css/style.css
    └── js/main.js
```

## Run locally

Python 3.10 or 3.11 is recommended for compatibility with the included scikit-learn model.

```bash
# 1. Unzip and enter the project folder
cd medical-ai-assistant

# 2. Create and activate a virtual environment
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

# 3. Install packages
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Start the app
python app.py
```

Open `http://127.0.0.1:5000`. Check `http://127.0.0.1:5000/api/health` for a JSON health response. The provided model files are included in the archive; if they are missing, run the dataset/model training scripts as described below.

## Optional database for history

The app runs without MongoDB. To persist the `/history` page, install/run MongoDB locally or create a free MongoDB Atlas cluster, then set `MONGO_URI` in your environment or `.env`. Never commit real credentials to GitHub. The free tier, quotas, and availability of third-party services can change.

## Free API options (optional)

You do **not** need an external API for the current version. If you later want to add an LLM explanation, providers sometimes offer a free developer tier (for example Google AI Studio Gemini or Groq); availability, model access, rate limits, and terms change, so check their current pricing and quotas before integrating. A free tier is not a guarantee of unlimited or permanently free usage. Keep any key on the backend in environment variables, never in frontend JavaScript.

For nearby facilities, the current version simply opens map search links, so no key is needed. OpenStreetMap's public data/services have usage policies; if you later query their APIs directly, follow their attribution and fair-use requirements.

## API contract

### `POST /predict`
Request JSON:

```json
{
  "symptoms": "I have headache and fever",
  "latitude": null,
  "longitude": null
}
```

`latitude` and `longitude` are optional. Response includes `predicted_disease`, `confidence`, `top3_predictions`, matched/negated symptoms, specialist, urgency, nearby-care search links, and general advice. The confidence is the model's score, not the probability that a real diagnosis is correct.

### `GET /api/health`
Returns basic backend status and confirms no external API key is required.

## Deploying

Deploy the Flask app on a Python-compatible host and configure its start command as `python app.py` (or use the host's documented WSGI command). Set the host-provided `PORT` environment variable if required. Free hosting plans may sleep, have resource limits, or change their terms. Ensure the model `.pkl` files are included in the deployed project. Do not enable Flask debug mode in production.

## Model limitations

The model predicts from a fixed training dataset and a limited symptom vocabulary. It can misclassify symptoms, and confidence scores can be misleading. Do not present results as confirmed disease, personalized treatment, or a substitute for a qualified clinician. Review the dataset and model quality before any demonstration.
