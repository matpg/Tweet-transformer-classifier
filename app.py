import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# title and description
st.write("""
# Tweet Transformer Classifier
Write your Tweet here:
""")

# search bar
query = st.text_input("Classify!", "")

@st.experimental_singleton
def credentials():
    ACCES_TOKEN = st.secrets["ACCESS_TOKEN"]
    MODEL_NAME =  st.secrets["MODEL_NAME"]
    return ACCES_TOKEN, MODEL_NAME


labels = ['agradecimiento', 'consulta', 'fraude', 'otro', 'problema',  'reclamo']

access_token, model_name = credentials()

@st.experimental_singleton
def transformer_model(access_token, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, user_token=access_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,  user_token=access_token)
    return tokenizer, model

tokenizer, model = transformer_model()

if query != "":
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model(**inputs)
    response = outputs.logits 
    response = response.detach().numpy()
    response = response.tolist()
    response = response[0]
    response = response.index(max(response))
    response = labels[response]
    st.write(f"Label: {response}")
