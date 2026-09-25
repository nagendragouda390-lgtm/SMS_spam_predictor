from flask import Flask, request, render_template
import joblib
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#-----------------------------

#NLTK setup

#-----------------------------

nltk.download("stopwords", quiet=True)

eng = set(stopwords.words("english"))
ps = PorterStemmer()

#-----------------------------

#Text preprocessing

#-----------------------------

def clean_text(text):
    text = text.lower()

    text = text.translate(
    str.maketrans("", "", string.punctuation)
)

    words = text.split()

    words = [w for w in words if w not in eng]

    words = [ps.stem(w) for w in words]

    return " ".join(words)

#-----------------------------

#Load complete ML pipeline

#-----------------------------

model = joblib.load("models/pipe.pkl")

#-----------------------------

#Flask app

#-----------------------------

app = Flask(name)

@app.route("/", methods=["GET", "POST"])
def home():

prediction = None
message = ""
prob = None

if request.method == "POST":

    message = request.form.get("message", "").strip()

    if message:

        # Same preprocessing used during training
        cleaned_message = clean_text(message)

        # Pipeline handles TF-IDF + model
        prediction = model.predict(
            [cleaned_message]
        )[0]

        # Probability of class 1 (Spam)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(
                [cleaned_message]
            )[0][1]

return render_template(
    "index.html",
    prediction=prediction,
    message=message,
    prob=prob
)

#-----------------------------

#Start Flask

#-----------------------------

if name == "main":
app.run(
host="0.0.0.0",
port=10000
)
