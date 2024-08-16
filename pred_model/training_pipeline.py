from pred_model.config import config  
from pred_model.processing.data_handling import load_dataset,save_pipeline
import pred_model.pipeline as pipe 
import os
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
datapath = root/"data"/"playground-series-s4e8"

# Define the paths
train_csv_path = os.path.join(datapath, config.TRAIN_FILE)
zip_file_path = os.path.join(datapath, config.ZIP_FILE)
zip_file_name = Path(config.ZIP_FILE).stem
save_model_path = os.path.join(root,'trained_models')

# Check if train.csv exists
if not os.path.exists(train_csv_path):
    print(f"'{config.TRAIN_FILE}' not found. Unzipping '{config.ZIP_FILE}'...")
    
    # Check if the zip file exists
    if os.path.exists(zip_file_path):
        # Unzip the file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall()  
        train_csv_path = os.path.join(datapath, zip_file_name, config.TRAIN_FILE)    
        print(f"'{config.ZIP_FILE}' has been unzipped.")
    else:
        print(f"'{config.ZIP_FILE}' not found.")
else:
    print(f"'{config.TRAIN_FILE}' found.")
    

def perform_training():
    train_X = load_dataset(train_csv_path)
    train_y = train_X[config.TARGET].map({'e':0,'p':1})
    pipe.classification_pipeline.fit(train_X[config.FEATURES], train_y)
    save_pipeline(pipe.classification_pipeline, save_model_path)

if __name__=='__main__':
    perform_training()

    
