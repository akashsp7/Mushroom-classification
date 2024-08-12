import os
import sys
import pandas as pd
import joblib

PACKAGE_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(str(PACKAGE_ROOT))

from pred_model.config import config

#Load the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH,file_name)
    _data = pd.read_csv(filepath)
    return _data

#Serialization
def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under the name {config.MODEL_NAME}")

#Deserialization
def load_pipeline(pipeline_to_load):
    load_path = os.path.join(config.SAVE_MODEL_PATH, pipeline_to_load)
    model_loaded = joblib.load(load_path)
    print(f"Model {pipeline_to_load} has been loaded")
    return model_loaded