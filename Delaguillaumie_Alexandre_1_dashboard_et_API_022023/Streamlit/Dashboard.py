#-----------------------#
# IMPORT LIBRARIES #
#-----------------------#

import streamlit as st
import ast
import joblib
import plotly.graph_objects as go
import matplotlib as plt
import plotly.express as px
import shap
import requests as re
import numpy as np
import plotly.express as px
import warnings
from PIL import Image

# ====================================================================
# HEADER - TITRE
# ====================================================================
html_header = """
<head>
<title>Application Dashboard credit Score</title>
<meta charset="utf-8">
<meta name="keywords" content="Home Crédit Group ( Dashboard, prêt, crédit score")>
<meta name="description" content="Application de Crédit Score">
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<h1 style="font-size:300%; color:#0031CC; font-family:Arial"> Prêt à dépenser <br>
<h2 style="color:GREY; font-family:Georgia"> DASHBOARD</h2>
<hr style= " display: block;
margin-top: 0;
margin-bottom: 0;
margin-left: auto;
margin-right: auto;
border-style: inset;
border-width: 1.5px;"/>
</h1>
"""

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<style>body{background-color: #fbfff0}</style>", 
            unsafe_allow_html=True)
st.markdown(html_header, unsafe_allow_html=True)

# Hide warning messages
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

#---------------------#
# STATIC VARIABLES #
#---------------------#

API_URL = "http://127.0.0.1:8010"

# Load data and models
@st.cache(allow_output_mutation=True)
def load_all_files():
    return joblib.load('all_data.pkl')

loaded_objects  = load_all_files()
data = loaded_objects['data']
infos_client = loaded_objects['infos_client']
pret_client = loaded_objects['pret_client']
preprocessed_data = loaded_objects['preprocessed_data']
model = loaded_objects['model']

# Extract column names and other required values
column_names = preprocessed_data.columns.tolist()

# Extract necessary steps from the model pipeline
classifier = model.named_steps['classifier']
df_preprocess = model.named_steps['preprocessor'].transform(data)
explainer = shap.TreeExplainer(classifier)
generic_shap = explainer.shap_values(df_preprocess, check_additivity=False)

# --------------------------------------------------------------------
# LOGO
# --------------------------------------------------------------------
# Chargement du logo de l'entreprise
logo = Image.open("logo.png")
st.sidebar.image(logo, width=240, 
                 caption=" Dashboard - Decision support tool",
                 use_column_width='always')


# Display the heading
st.title("Loan Approval Dashboard")
st.markdown("Make informed decisions about loan approvals")

# Profile Client
with st.sidebar:
    st.markdown("### Profile Client")
    profile_ID = st.selectbox('Select a client:', list(data.index))
    API_GET = API_URL+"/predict/"+(str(profile_ID))
    score_client = 100 - int(re.get(API_GET).json() * 100)

    # Check if the client is eligible for a loan based on the score
    if score_client < 100 - 10.344827586206896:
        st.error("Loan Denied")
    else:
        st.success("Loan Approved")

    # Display the gauge
    gauge_figure = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=score_client,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [None, 100],
                        'tickwidth': 3,
                        'tickcolor': 'gray'},
                'bar': {'color': '#F0F2F6', 'thickness': 0.6},
                'steps': [{'range': [0, 50], 'color': 'red'},
                          {'range': [50, 70], 'color': 'orange'},
                          {'range': [70, 89], 'color': 'gold'},
                          {'range': [89, 95], 'color': 'limegreen'},
                          {'range': [95, 100], 'color': 'green'}]}))
    gauge_figure.update_layout(height=250, width=450, margin=dict(t=80, b=0))
    st.plotly_chart(gauge_figure)

# Choose between displaying client information or prediction details
client_info_checkbox = st.checkbox('Client Info')
clinet_comp_checkbox = st.checkbox('Client Comparison')
client_pred_checkbox = st.checkbox('Prediction',value=True)

if client_info_checkbox:
    client_info = infos_client[infos_client.index == profile_ID].iloc[:, :]
    client_info_dict = client_info.to_dict('list')
    st.markdown('**Client Info:**')
    for i, j in client_info_dict.items():
        st.text(f"{i} = {j[0]}")
        
if clinet_comp_checkbox:
    st.markdown('**Client Comparison:**')
    features_to_compare = st.multiselect('Select features to compare', list(data.columns))
    for feature in features_to_compare:
        st.text(f'Feature = {feature}')
        fig1 = px.histogram(data, x=feature)
        st.text('Others')
        st.plotly_chart(fig1)
        fig2 = px.histogram(data[data.index == profile_ID], x=feature)
        st.text('Client')
        st.plotly_chart(fig2)

        
if client_pred_checkbox:
    st.markdown('**Client Prediction:**')
    if 95 <= score_client < 100:
        score_text = 'PERFECT LOAN APPLICATION'
        st.success(score_text)
    elif 100 - 10.344827586206896 <= score_client < 95:
        score_text = 'GOOD LOAN APPLICATION'
        st.success(score_text)
    elif 70 <= score_client < 100 - 10.344827586206896:
        score_text = 'REVIEW REQUIRED'
        st.warning(score_text)
    else:
        score_text = 'INSOLVENT LOAN APPLICATION'
        st.error(score_text)

    # Display loan details for the selected client
    st.subheader("Loan Details")
    client_pret = pret_client[pret_client.index == profile_ID].iloc[:, :]
    st.table(client_pret)

    # Contribution of variables to the model
    st.subheader("Variable Contributions to the Model")

    col1, col2 = st.columns([4, 3.3])
    with col1:
        # Local interpretability using SHAP
        st.write(f"For client {profile_ID}:")
        API_GET = API_URL+"/shap_client/" + (str(profile_ID))
        shap_values = re.get(API_GET).json()
        shap_values = ast.literal_eval(shap_values['shap_client'])
        shap_values = np.array(shap_values).astype('float32')
        waterfall = shap.plots._waterfall.waterfall_legacy(shap_values=shap_values,
                                                           expected_value = -2.9159221699244515,
                                                           feature_names=column_names,
                                                           max_display=20)
        st.pyplot(waterfall)

    with col2:
        # Global interpretability using SHAP
        st.write("For all clients:")
        summary = shap.summary_plot(shap_values=generic_shap,
                                    feature_names=column_names,
                                    max_display=20)
        st.pyplot(waterfall)

    # Interactive graphs
    st.subheader("Interactive Classification Graph")
    features = st.multiselect("Choose two variables",
                              list(data.columns),
                              default=['AMT_ANNUITY', 'AMT_INCOME_TOTAL'],
                              max_selections=2)
    if len(features) != 2:
        st.error("Select two variables")
    else:
        # Plot the graph
        chart = px.scatter(data,
                           x=features[0],
                           y=features[1],
                           color='TARGET',
                           color_discrete_sequence=['limegreen', 'tomato'],
                           hover_name=data.index)
        st.plotly_chart(chart)
