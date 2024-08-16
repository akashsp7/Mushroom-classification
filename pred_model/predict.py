import pandas as pd
import numpy as np
import os
from pathlib import Path
from pred_model.config import config  
from pred_model.processing.data_handling import load_pipeline, load_dataset
import zipfile

root = Path(__file__).resolve().parent
datapath = root/"data"/"playground-series-s4e8"

test_csv_path = os.path.join(datapath, config.TEST_FILE)
zip_file_path = os.path.join(datapath, config.ZIP_FILE)
zip_file_name = Path(config.ZIP_FILE).stem
save_results_path = os.path.join(root,'results')
load_model_path = os.path.join(root,'trained_models')

# Check if train.csv exists
if not os.path.exists(test_csv_path):
    print(f"'{config.TEST_FILE}' not found. Unzipping '{config.ZIP_FILE}'...")
    
    # Check if the zip file exists
    if os.path.exists(zip_file_path):
        # Unzip the file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(datapath)  
        print(f"'{config.ZIP_FILE}' has been unzipped.")
    else:
        print(f"'{config.ZIP_FILE}' not found.")
else:
    print(f"'{config.TEST_FILE}' found.")

classification_pipeline, model_name = load_pipeline(load_model_path)

def generate_predictions(data_input):
    data = load_dataset(data_input)
    ids = data['id'].to_numpy()
    pred = classification_pipeline.predict(data[config.FEATURES])
    output = np.where(pred==1,'p','e')
    result = pd.DataFrame({'id': ids,
                           'class': output})
    result.to_csv(os.path.join(save_results_path, f'results-{model_name}.csv'), index=False)
    print(f'results-{model_name}.csv saved at {save_results_path}')
    return result

if __name__=='__main__':
    generate_predictions(test_csv_path)
