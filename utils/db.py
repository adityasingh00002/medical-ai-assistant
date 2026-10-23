"""
db.py
-----
MongoDB integration. Stores every symptom-check interaction so the app
has a "consultation history" feature and you have real data to show in
your project demo/report.

Setup:
  Option A (local): install MongoDB Community Server, run `mongod`,
                     leave MONGO_URI unset (defaults to localhost).
  Option B (cloud, recommended for a project you'll demo anywhere):
                     create a free MongoDB Atlas cluster, then set:
                     MONGO_URI=mongodb+srv://<user>:<pass>@cluster0.xxxx.mongodb.net/

Collection: medical_ai.consultations
Document shape:
{
  "_id": ObjectId,
  "timestamp": ISO datetime string,
  "user_input": "raw text the user typed",
  "matched_symptoms": [...],
  "negated_symptoms": [...],
  "predicted_disease": "...",
  "confidence": 0.83,
  "specialist": "...",
  "urgency": "soon",
  "advice": { ...llm output... }
}
"""

import os
import datetime

_client = None
_db = None


def get_db():
    """Lazily creates and caches a MongoDB connection. Returns None (with
    a printed warning) if MongoDB isn't reachable, so the rest of the app
    can still run in a degraded 'no history' mode during a demo."""
    global _client, _db
    if _db is not None:
        return _db

    try:
        from pymongo import MongoClient

        mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
        _client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        _client.server_info()  # force connection check
        _db = _client["medical_ai"]
        return _db
    except Exception as e:
        print(f"[db.py] WARNING: MongoDB not available ({e}). "
              f"Running without persistent history.")
        return None


def save_consultation(record: dict):
    """Insert one consultation record. Silently no-ops if DB unavailable."""
    db = get_db()
    if db is None:
        return None
    record["timestamp"] = datetime.datetime.utcnow().isoformat()
    result = db.consultations.insert_one(record)
    return str(result.inserted_id)


def get_recent_consultations(limit=20):
    """Fetch most recent consultations (for a /history page). Returns
    an empty list if DB unavailable."""
    db = get_db()
    if db is None:
        return []
    docs = list(
        db.consultations.find().sort("timestamp", -1).limit(limit)
    )
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


if __name__ == "__main__":
    test_id = save_consultation({
        "user_input": "test entry from db.py __main__",
        "predicted_disease": "Common Cold",
        "confidence": 0.9,
    })
    print("Inserted id:", test_id)
    print("Recent:", get_recent_consultations(5))
