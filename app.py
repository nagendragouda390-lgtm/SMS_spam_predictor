from flask import Flask, request, render_template
import joblib
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# -----------------------------
# NLTK setup
# -----------------------------
nltk.download("stopwords", quiet=True)

eng = set(stopwords.words("english"))
ps = PorterStemmer()


# -----------------------------
# Text preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [w for w in words if w not in eng]

    words = [ps.stem(w) for w in words]

    return " ".join(words)


# -----------------------------
# Load complete ML pipeline
# -----------------------------
model = joblib.load("models/pipe.pkl")


# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    message = ""

    if request.method == "POST":

        message = request.form.get("message", "")

        if message.strip():

            # Same preprocessing used during training
            cleaned_message = clean_text(message)

            # Pipeline handles TF-IDF + model
            prediction = model.predict(
                [cleaned_message]
            )[0]

            prob = model.predict_proba([cleaned_message])[0][1]

    return render_template(
        "index.html",
        prediction=prediction,
        message=message,
        prob = prob
    )


# -----------------------------
# Start Flask
# -----------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000
    )
