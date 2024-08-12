import pandas as pd
import numpy as np 
from pathlib import Path
import os
import sys

PACKAGE_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(str(PACKAGE_ROOT))

from pred_model.config import config  
from pred_model.processing.data_handling import load_dataset,save_pipeline
import pred_model.pipeline as pipe 

def perform_training():
    train_X = load_dataset(config.TRAIN_FILE)
    train_y = train_X[config.TARGET].map({'e':0,'p':1})
    pipe.classification_pipeline.fit(train_X[config.FEATURES], train_y)
    save_pipeline(pipe.classification_pipeline)

if __name__=='__main__':
    perform_training()
        