# save_model.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

nltk.download('stopwords')

# Load Dataset
data = pd.read_csv("cyberbullying_tweets.csv")

# Simplify labels
data['label'] = data['cyberbullying_type'].apply(
    lambda x: 'not_cyberbullying' if x == 'not_cyberbullying' else 'cyberbullying'
)


# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z0-9\s😊😂❤️👍🙏]', '', text)
    return text.strip()

data['clean_text'] = data['tweet_text'].apply(clean_text)

# Split
X = data['clean_text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
tfidf = TfidfVectorizer(stop_words=stopwords.words('english'), max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)

# Train Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
joblib.dump(model, "cyberbully_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("✅ Model and TF-IDF saved successfully!")
