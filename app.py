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
        text += '{}: {}\n'.format(col.upper(), results[0][idx] > 0.5)

    return text

model = tf.keras.models.load_model(r'toxicity.h5')
vmodel = tf.keras.models.load_model(r"vectorizer")
vectorizer = vmodel.layers[0]

st.title('Comment Toxicity Calculator')
st.write("Source Code -")
st.hyperlink("https://github.com/WhiteWolf47/Comment_toxicity")
#st.image("https://wallpapercave.com/dwp1x/wp10935197.jpg")
user_input = ""
user_input = st.text_area("Enter Comment")

if st.button('Calculate Toxicicty'):
    if user_input == "":
        st.title("Please enter a valid comment")
    else:
        res = score_comment(user_input)
        st.text(res)
