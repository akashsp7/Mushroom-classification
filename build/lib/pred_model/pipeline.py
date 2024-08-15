
import os
import sys

PACKAGE_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(str(PACKAGE_ROOT))

from sklearn.pipeline import Pipeline
import numpy as np
from pred_model.config import config
import pred_model.processing.preprocessing as pp 
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

classification_pipeline = Pipeline(
    [
        ('DropFeatures', pp.DropColumns(features=config.DROP_FEATURES)),
        ('CatImputer',pp.CatImputer(features=config.CAT_FEATURES)),
        ('NumImputer',pp.NumImputer(features=config.NUM_IMPUTE)),
        ('AddFeatures', pp.AddFeatures(added_features=config.ADDED_FEATURES)),
        ('LogTransform',pp.LogTransforms(features=config.NUM_FEATURES)),
        ('OE', pp.CustomOrdinalEncoder(features=config.CAT_FEATURES)),
        ('MinMaxScale', MinMaxScaler()),
        ('Classifier', XGBClassifier(**config.PARAMS))
    ]
)