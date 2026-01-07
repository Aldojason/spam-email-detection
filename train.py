import pandas as pd
import string
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

# Load dataset
data = pd.read_csv("data/spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Clean text
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

data['clean_message'] = data['message'].apply(clean_text)

# Encode labels
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data['clean_message'])
y = data['label_num']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model accuracy:", accuracy)

# Save model & vectorizer
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model and vectorizer saved successfully!")
