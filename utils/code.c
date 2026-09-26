import pandas as pd
import numpy as np
import seaborn as sns
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report as cr

import joblib
# importing stopwords of english
eng = set(stopwords.words("english"))

df = pd.read_csv("data/sms_spam.csv",encoding="latin1")

df = df.drop_duplicates()

df["label"] = df["label"].map({"ham":0,"spam":1})

def clean_text(text):
    text = text.lower()
    
    text = text.translate(str.maketrans("","",string.punctuation))
    
    word = text.split()
    
    word = [w for w in word if w not in eng]
    
    ps = PorterStemmer()
    
    word = [ps.stem(w) for w in word]
    
    return " ".join(word)
    
df["cleaned"] = df["message"].apply(clean_text)

X = df["cleaned"]
y = df["label"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

pipe = Pipeline([
    ("tfidf",TfidfVectorizer(max_features=3000)),
    ("model",MultinomialNB())])   

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print(cr(y_test,y_pred))

joblib.dump(pipe,"models/pipe.pkl")


