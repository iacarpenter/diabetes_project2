import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def load_csv_data(csv_file_path, separator):
    return pd.read_csv(csv_file_path, sep=separator)

def stratified_split_dataset(
        data, test_size, random_state, 
        stratification_feature, label_feature):
    
    split = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state)

    # shuffle and subdivide dataset into train and test sets, maintaining 
    # the stratification feature proportions of the original dataset
    for train_index, test_index in split.split(data, data[stratification_feature]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    # split the labels off of both datasets (and reset their indexes)
    train_labels = strat_train_set[label_feature].copy().reset_index(drop=True)
    strat_train_set = strat_train_set.drop(label_feature, axis=1).reset_index(drop=True)
    test_labels = strat_test_set[label_feature].copy().reset_index(drop=True)
    strat_test_set = strat_test_set.drop(label_feature, axis=1).reset_index(drop=True)

    return strat_train_set, train_labels, strat_test_set, test_labels

