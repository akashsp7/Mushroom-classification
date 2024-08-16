import pandas as pd
import numpy as np
import os
from pathlib import Path
from pred_model.config import config  
from pred_model.processing.data_handling import load_pipeline, load_dataset

ROOT = Path(__file__).resolve().parent
TEST_FILE = 'test.csv'
SAVE_RESULTS_PATH = os.path.join(ROOT,'results')

classification_pipeline, model_name = load_pipeline(config.MODEL_NAME)

def generate_predictions(data_input):
    data = load_dataset(data_input)
    ids = data['id'].to_numpy()
    pred = classification_pipeline.predict(data[config.FEATURES])
    output = np.where(pred==1,'p','e')
    result = pd.DataFrame({'id': ids,
                           'class': output})
    result.to_csv(os.path.join(SAVE_RESULTS_PATH, f'results-{model_name}.csv'), index=False)
    print(f'results-{model_name}.csv saved at {SAVE_RESULTS_PATH}')
    return result

if __name__=='__main__':
    generate_predictions(TEST_FILE)
