import os
import joblib
import pandas as pd
import pickle

def test_loaded_files():
    all_files_path = os.path.join('..','..','Delaguillaumie_Alexandre_1_dashboard_et_API_022023','Streamlit','all_data.pkl') 
    data_required = ['data', 'infos_client', 'pret_client', 'preprocessed_data', 'model']
    loaded_objects = joblib.load(all_files_path)
    for i in data_required:
        assert i in list(loaded_objects.keys())
        
def test_duplicates():
    all_files_path = os.path.join('..','..','Delaguillaumie_Alexandre_1_dashboard_et_API_022023','Streamlit','all_data.pkl') 
    loaded_objects = joblib.load(all_files_path)
    df = loaded_objects['preprocessed_data']
    assert df.duplicated().sum() == 0
    
def test_null_values():
    all_files_path = os.path.join('..','..','Delaguillaumie_Alexandre_1_dashboard_et_API_022023','Streamlit','all_data.pkl') 
    loaded_objects = joblib.load(all_files_path)
    df = loaded_objects['preprocessed_data']
    assert df.isna().sum().sum() == 0
    
def test_column_names():
    with open(os.path.join('..','..','data','sample_data.pkl'),'rb') as f:
        df1 = pickle.load(f)
        
    df2 = pd.read_csv(os.path.join('..','..', 'data', 'preprocessed_data.csv'))
    col1 = df1.columns.tolist()
    cnt = 0
    for i in col1:
        if i not in df2.columns:
            cnt+=1
    assert cnt == 0
    
def test_predict_positive_class():
    all_files_path = os.path.join('..','..','Delaguillaumie_Alexandre_1_dashboard_et_API_022023','Streamlit','all_data.pkl') 
    loaded_objects = joblib.load(all_files_path)
    model = loaded_objects['model']

    df = pd.read_csv(os.path.join('..','..', 'data', 'preprocessed_data.csv'))
    outcome = model.predict(df[df.index == 19171])[0]
    assert outcome == 1
    
def test_predict_negative_class():
    all_files_path = os.path.join('..','..','Delaguillaumie_Alexandre_1_dashboard_et_API_022023','Streamlit','all_data.pkl') 
    loaded_objects = joblib.load(all_files_path)
    model = loaded_objects['model']

    df = pd.read_csv(os.path.join('..','..', 'data', 'preprocessed_data.csv'))
    outcome = model.predict(df[df.index == 0])[0]
    assert outcome == 0