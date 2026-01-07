from flask import Flask, request, jsonify
from flask import render_template
from flask import send_from_directory
import joblib
import string
import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Load saved model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Text cleaning function (same as training)
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message")

    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    result = "Spam" if prediction == 1 else "Not Spam"

    return jsonify({"prediction": result})

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        'static',
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )


if __name__ == "__main__":
    app.run(debug=True)
