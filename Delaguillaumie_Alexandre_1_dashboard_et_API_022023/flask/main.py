import warnings
warnings.filterwarnings('ignore')

import joblib
import shap
import json
from flask import Flask, request, jsonify

# Create the Flask app object
app = Flask(__name__)

# Load the model and data
model = joblib.load('model.pkl')
data = joblib.load('sample_test_set.pickle')
client_ids = data.index.tolist()

# Retrieve the classifier from the model pipeline
classifier = model.named_steps['classifier']

@app.route("/predict", methods=["POST"])
def predict():
    d = request.json
    client_id = d.get("client_id")
    
    # Perform predictions and return the probability of positive class
    predictions = model.predict_proba(data).tolist()
    predict_proba = []
    for pred, ID in zip(predictions, client_ids):
        if ID == client_id:
            predict_proba.append(pred[1])
    return jsonify({"predict_proba": predict_proba[0]})

@app.route('/generic_shap')
def generic_shap():
    # Calculate SHAP values for all clients
    df_preprocess = model.named_steps['preprocessor'].transform(data)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(df_preprocess, check_additivity=False)
    shap_values_list = [value.tolist() for value in shap_values]
    json_shap = json.dumps(shap_values_list)
    return {'shap_values': json_shap}


@app.route("/shap_client", methods=["POST"])
def shap_client():
    d = request.json
    client_id = d.get("client_id")
    
    # Calculate SHAP values for a specific client
    index_id = []
    for ind, ID in enumerate(client_ids):
        if client_ids[ind] == client_id:
            index_id.append(ind)
        else:
            pass
    df_preprocess = model.named_steps['preprocessor'].transform(data)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(df_preprocess, check_additivity=False)
    shap_values_client = shap_values[index_id][0]
    json_shap_client = json.dumps(shap_values_client.tolist())
    return {'shap_client': json_shap_client}

# Run the Flask API
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8010)
    # app.run(debug=True, host='35.180.29.152', port=8000)
