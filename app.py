import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import distance

# bootstrap style
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
""", unsafe_allow_html=True)

# title and description
st.title("#Tweet Transformer Classifier")
st.markdown("Hugging Face fine tuned AutoNLP model is used to classify tweets")
st.write("Write your Tweet here:")


# search bar
query = st.text_input("", "")

# Classify button
btn = st.button("Classify!")


@st.experimental_singleton
def credentials():
    ACCES_TOKEN = st.secrets["ACCESS_TOKEN"]
    MODEL_NAME =  st.secrets["MODEL_NAME"]
    return ACCES_TOKEN, MODEL_NAME


labels = ['agradecimiento', 'consulta', 'fraude', 'otro', 'problema',  'reclamo']

access_token, model_name = credentials()

# Load transformer model
@st.experimental_singleton
def transformer_model(access_token, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,  use_auth_token=access_token)
    return tokenizer, model

tokenizer, model = transformer_model(access_token, model_name)

# analyze category of tweet
@st.cache
def CategoriesAnalyze(tweet):
    # for each word in categories, verify if it is in the tweet
    categories = [
        {'Cluster Word': 'aplicación', 'Candidates': ['aplicación', 'actualizacion', 'aplicacion', 'app', 'plataforma', 'notificación']},
        {'Cluster Word': 'transferencias', 'Candidates': ['transferencia', 'transferencias', 'pago', 'cuenta','cuentas', 'transferir', 'transacción']},
        {'Cluster Word': 'banca', 'Candidates': ['banca', 'banco', 'cuentas', 'linea', 'en linea', 'cuenta']},
        {'Cluster Word': 'teléfono', 'Candidates': ['teléfono', 'telefono', 'numero', 'telefonos', 'celular', 'celulares']},
        {'Cluster Word': 'bepass', 'Candidates': ['bepass', 'clave', 'pass']},
        {'Cluster Word': 'pago', 'Candidates': ['pago', 'pagos', 'cuentas', 'cuenta', 'cuentas']},
        {'Cluster Word': 'cajero', 'Candidates': ['cajero', 'cajeros', 'cajas', 'caja', 'sistema']},
        {'Cluster Word': 'página', 'Candidates': ['pagina', 'página' 'sitio', 'web', 'paginas', 'sitios', 'paginas', 'portal', 'empresa', 'plataforma']},
        {'Cluster Word': 'tarjeta', 'Candidates': ['tarjeta', 'tarjetas']}
    ]
    affinity_words, keyword_category = np.empty(0), np.empty(0)
    tweet = tweet.split()
    for category in categories:
        for candidate in category['Candidates']:
            for word in tweet:
                affinity = distance.levenshtein(candidate, word, max_dist=2)
                affinity_words = np.append(
                    affinity_words, {'Candidate': candidate, 'Keyword': category['Cluster Word'], 'Affinity': affinity}
                    )

    for aff_dict in affinity_words:
        if -1 < aff_dict['Affinity'] <= 1:
            # save only the affinity words with affinity 0 and 1, (0 diff and only 1 diff)
            keyword_category = np.append(
                keyword_category, {'Aff word': aff_dict['Candidate'], 'Keyword': aff_dict['Keyword'], 'Affinity': aff_dict['Affinity']}
                )
    if keyword_category.any():
        keyword_category = sorted(keyword_category, key=lambda k: k['Affinity'], reverse=True)
        keyword_list = [w['Keyword'] for w in keyword_category]
        # delete duplicates values from the list keyword_list
        return list(np.unique(keyword_list))
    else:
        return 'Category Not Found'

if query != "" or btn:
    st.balloons()
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model(**inputs)
    response = outputs.logits 
    response = response.detach().numpy()
    response = response.tolist()
    response = response[0]
    response = response.index(max(response))
    response = labels[response]
    st.write(f"Label: {response}")
    if response == "problema":
        category = CategoriesAnalyze(query)
        categories = ", ".join([cat for cat in category])
        st.write(f"Categories: {categories}")

