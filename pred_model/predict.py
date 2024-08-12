import pandas as pd
import numpy as np

import os
import sys

PACKAGE_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(str(PACKAGE_ROOT))

from pred_model.config import config  
from pred_model.processing.data_handling import load_pipeline, load_dataset

classification_pipeline = load_pipeline(config.MODEL_NAME)

def generate_predictions(data_input):
    data = load_dataset(data_input)
    ids = data['id'].to_numpy()
    pred = classification_pipeline.predict(data[config.FEATURES])
    output = np.where(pred==1,'p','e')
    result = pd.DataFrame({'id': ids,
                           'class': output})
    result.to_csv(os.path.join(config.SAVE_RESULTS_PATH, 'results.csv'), index=False)
    print(f'results.csv saved at {config.SAVE_RESULTS_PATH}')
    return result

if __name__=='__main__':
    generate_predictions(config.TEST_FILE)
