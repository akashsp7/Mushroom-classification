from sklearn.base import BaseEstimator,TransformerMixin
import os
import sys

PACKAGE_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(str(PACKAGE_ROOT))

from pred_model.config import config
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

class DropColumns(BaseEstimator,TransformerMixin):
    def __init__(self,features=None):
        self.features = features
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        X = X.drop(columns = self.features)
        return X
    
class AddFeatures(BaseEstimator,TransformerMixin):
    def __init__(self, added_features=None):
        self.added_features = added_features
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        if 'cap-area' in self.added_features:
            X['cap-area'] = np.pi * (X['cap-diameter']/2) ** 2
        if 'stem-volume' in self.added_features:
            X['stem-volume'] = np.pi * (X['stem-width'] / 2) ** 2 * X['stem-height']
        return X

class LogTransforms(BaseEstimator,TransformerMixin):
    def __init__(self,features=None):
        self.features = features
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        for col in self.features:
            X[col] = np.log1p(X[col])
        return X
    
class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    def fit(self, X, y=None):
        self.encoder.fit(X[self.features])
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.features] = self.encoder.transform(X[self.features])
        return X
    
class NumImputer(BaseEstimator,TransformerMixin):
    def __init__(self,features=None):
        self.features = features
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    def fit(self,X,y=None):
        self.imputer.fit(X[self.features])
        return self
    
    def transform(self,X):
        X = X.copy()
        X[self.features] = self.imputer.transform(X[self.features])
        return X
    
class CatImputer(BaseEstimator,TransformerMixin):
    def __init__(self,features=None):
        self.features = features
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    
    def fit(self,X,y=None):
        self.imputer.fit(X[self.features])
        return self
    
    def transform(self,X):
        X = X.copy()
        X[self.features] = self.imputer.transform(X[self.features])
        return X
    