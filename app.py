import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

t_params = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
       'identity_hate']

def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)

    text = ''
    for idx, col in enumerate(t_params):
        text += '{}: {}\n'.format(col, results[0][idx] > 0.5)

    return text

model = tf.keras.models.load_model(r'C:\Users\anura\PycharmProjects\pythonProject\venv\toxicity.h5')
vmodel = tf.keras.models.load_model(r"C:\Users\anura\PycharmProjects\pythonProject\venv\vectorizer")
vectorizer = vmodel.layers[0]

st.title('Comment Toxicity Calculator')
#st.image("https://wallpapercave.com/dwp1x/wp10935197.jpg")
user_input = ""
user_input = st.text_area("Enter Comment")

if st.button('Calculate Toxicicty'):
    if user_input == "":
        st.text("Please enter a valid comment")
    else:
        st.title('Toxic or non-Toxic')
        res = score_comment(user_input)
        st.text(res)