import streamlit as st
import pickle

pipe = pickle.load(,"rb")

st.title("SMS spam predictor")

message = st.text_area("enter sms")

if st.button("Predict"):
  if message.strip() == "":
    st.warning("Please enter message ")
  else:
    pred = pipe.predict(message)[0]
    prob = pipe.predict_proba(message)[0][1]

    if pred == 1:
      st.error("Spam !")
    else:
      st.success("Not spam !\n\n Prob : {prob}")
  
