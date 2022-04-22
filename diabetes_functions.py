from basic_functions import load_csv_data, stratified_split_dataset
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "./diabetes_012_health_indicators_BRFSS2015.csv"
SEPARATOR = ","

def load_diabetes_data(dataset_path=DATASET_PATH, separator=SEPARATOR):
    return load_csv_data(dataset_path, separator)

def split_diabetes_dataset(
        diabetes_data, test_size=0.2, random_state=42, 
        stratification_feature="Diabetes_012", 
        label_feature="Diabetes_012"):
    
    return stratified_split_dataset(
        diabetes_data, test_size, random_state, 
        stratification_feature, label_feature)

class CustomAttributeDropper (BaseEstimator, TransformerMixin):

    # transformer to drop PhysHlth, DiffWalk, and Education

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        physhlth_ix = 15
        diffwalk_ix = 16
        education_ix = 19
        X = np.delete(X, [physhlth_ix, diffwalk_ix, education_ix], 1)
        return X

# pipeline to drop features, then scale