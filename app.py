import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# title and description
st.write("""
# Tweet Transformer Classifier
Write your Tweet here:
""")

# search bar
query = st.text_input("Classify!", "")

labels = ['agradecimiento', 'consulta', 'fraude', 'otro', 'problema',  'reclamo']

access_token = st.secrets["ACCESS_TOKEN"]

MODEL_NAME = st.secrets["MODEL_NAME"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=access_token)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, use_auth_token=access_token)

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
