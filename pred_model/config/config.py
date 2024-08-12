import os
import numpy as np
from pathlib import Path

ROOT = Path.cwd()

DATAPATH = ROOT/"data"/"playground-series-s4e8"

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

MODEL_NAME = 'mushroom_model.pkl'
SAVE_MODEL_PATH = os.path.join(ROOT,'trained_models')
SAVE_RESULTS_PATH = os.path.join(ROOT,'results')

TARGET = 'class'

#Initial features used in the model
FEATURES = ['id', 'cap-diameter', 'cap-shape', 'cap-surface', 'cap-color',
       'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
       'stem-height', 'stem-width', 'stem-root', 'stem-surface', 'stem-color',
       'veil-type', 'veil-color', 'has-ring', 'ring-type', 'spore-print-color',
       'habitat', 'season']

NUM_FEATURES = ['cap-diameter', 'stem-height', 'stem-width', 'cap-area', 'stem-volume']

NUM_IMPUTE = ['cap-diameter', 'stem-height', 'stem-width']

CAT_FEATURES = ['cap-shape','cap-color','does-bruise-or-bleed',
                'gill-color','stem-color','has-ring','ring-type',
                'habitat','season']

DROP_FEATURES = ['cap-surface','gill-attachment','gill-spacing',
                 'stem-root','stem-surface','veil-type','veil-color',
                 'spore-print-color', 'id']

ADDED_FEATURES = ['cap-area', 'stem-volume']

PARAMS = {'n_estimators': 936, 'max_leaves': 116,
                   'min_child_weight': np.float64(4.437590285996336),
                   'learning_rate': np.float64(0.24964403972620555),
                   'subsample': np.float64(0.8823329539295974),
                   'colsample_bylevel': np.float64(0.835473445927455),
                   'colsample_bytree': np.float64(0.707587058577376),
                   'reg_alpha': np.float64(0.005664534593409382),
                   'reg_lambda': np.float64(0.013343252707226935)}
