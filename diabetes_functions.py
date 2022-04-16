from basic_functions import load_csv_data, stratified_split_dataset

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

