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
import warnings

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide warning messages
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

#---------------------#
# STATIC VARIABLES #
#---------------------#

API_URL = "http://127.0.0.1:8010"
background_img_url = 'https://images.unsplash.com/photo-1501426026826-31c667bdf23d'

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("{background_img_url}");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Load data and models
@st.cache(allow_output_mutation=True)
def load_data():
    return joblib.load('sample_test_set.pickle')

@st.cache(allow_output_mutation=True)
def load_infos_client():
    return joblib.load('infos_client.pickle')

@st.cache(allow_output_mutation=True)
def load_pret_client():
    return joblib.load('pret_client.pickle')

@st.cache(allow_output_mutation=True)
def load_preprocessed_data():
    return joblib.load('preprocessed_data.pickle')

@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('model.pkl')

data = load_data()
infos_client = load_infos_client()
pret_client = load_pret_client()
preprocessed_data = load_preprocessed_data()
model = load_model()

# Extract column names and other required values
column_names = preprocessed_data.columns.tolist()

# Extract necessary steps from the model pipeline
classifier = model.named_steps['classifier']
df_preprocess = model.named_steps['preprocessor'].transform(data)
explainer = shap.TreeExplainer(classifier)
generic_shap = explainer.shap_values(df_preprocess, check_additivity=False)

# Set background image for the page
page_bg_img = f"""
<style>
body {{
    background-image: url("{background_img_url}");
    background-size: cover;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

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
c = st.selectbox('Select:', ['Client Info', 'Prediction'])
if c == 'Client Info':
    client_info = infos_client[infos_client.index == profile_ID].iloc[:, :]
    client_info_dict = client_info.to_dict('list')
    st.markdown('**Client Info:**')
    for i, j in client_info_dict.items():
        st.text(f"{i} = {j[0]}")
else:
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