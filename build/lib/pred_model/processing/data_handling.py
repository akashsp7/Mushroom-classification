import os
import pandas as pd
import joblib
from pred_model.config import config

#Load the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH,file_name)
    _data = pd.read_csv(filepath)
    return _data

#Serialization
def save_pipeline(pipeline_to_save):
    base_name = config.MODEL_NAME
    save_path = config.SAVE_MODEL_PATH
    
    existing_models = [f for f in os.listdir(save_path) if f.startswith(base_name)]
    version_numbers = [
        int(f.split('-v')[-1].split('.')[0]) for f in existing_models if '-v' in f
    ]
    next_version = max(version_numbers, default=0) + 1
    
    model_name_with_version = f"{base_name}-v{next_version}.pkl"
    full_save_path = os.path.join(save_path, model_name_with_version)
    
    joblib.dump(pipeline_to_save, full_save_path)
    
    print(f"Model has been saved under the name {model_name_with_version}")

#Deserialization
def load_pipeline():
    base_name = config.MODEL_NAME
    save_path = config.SAVE_MODEL_PATH
    existing_models = [f for f in os.listdir(save_path) if f.startswith(base_name)]
    version_numbers = [
        int(f.split('-v')[-1].split('.')[0]) for f in existing_models if '-v' in f
    ]
    latest_version = max(version_numbers, default=1)
    
    model_name_with_version = f"{base_name}-v{latest_version}.pkl"
    full_load_path = os.path.join(save_path, model_name_with_version)
    model_loaded = joblib.load(full_load_path)
    print(f"{model_name_with_version} has been loaded")
    return model_loaded
