import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# bootstrap style
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
""", unsafe_allow_html=True)

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
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,  use_auth_token=access_token)
    return tokenizer, model
tokenizer, model = transformer_model(access_token, model_name)

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


'''
def card(id_val, source, context):
    st.markdown(f"""
    <div class="card" style="margin:1rem;">
        <div class="card-body">
            <h5 class="card-title">{source}</h5>
            <h6 class="card-subtitle mb-2 text-muted">{id_val}</h6>
            <p class="card-text">{context}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
