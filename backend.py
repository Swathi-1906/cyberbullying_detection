# backend.py
from flask import Flask, request, jsonify
import joblib
import re
from textblob import TextBlob

app = Flask(__name__)

# Load saved model and vectorizer
model = joblib.load("cyberbully_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z0-9\s😊😂❤️👍🙏]', '', text)
    return text.strip()

def is_neutral_or_positive(text):
    text_lower = text.lower()
    blob = TextBlob(text_lower)
    sentiment = blob.sentiment.polarity  # range -1 to +1

    # Detect sarcasm or mixed emotion
    if ("happy" in text_lower or "glad" in text_lower) and ("sad" in text_lower or "cry" in text_lower or "fail" in text_lower):
        # Example: "I am happy because you are sad"
        return False  # treat as cyberbullying

    # Positive or neutral tone detection
    if sentiment >= 0.1:
        return True

    # Extra keyword-based check for politeness / gratitude
    positive_words = [
        "love", "sweet", "dear", "thank", "good", "nice", "beautiful", "happy",
        "great", "wonderful", "awesome", "kind", "appreciate", "help", "friend", "support","wonder","hi","how are you","where","when","why"
    ]
    negative_words = [
        "hate", "stupid", "ugly", "kill", "dumb", "idiot", "disgusting", "loser", "sad","ruining","screwed"
    ]

    if any(word in text_lower for word in positive_words) and not any(word in text_lower for word in negative_words):
        return True
    return False


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    confidence = round(max(proba) * 100, 2)

    if is_neutral_or_positive(text):
        prediction = "not_cyberbullying"

    return jsonify({
        "prediction": prediction,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)