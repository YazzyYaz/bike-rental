import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import dataset

class Dataset(object):
    """Dataset Class to instantiate test and training datasets for bike"""
    def __init__(self, data_url):
        self.data_url = data_url
        self.dummy_fields = dataset.get('dummy_fields')
        self.fields_to_drop = dataset.get('fields_to_drop')
        self.quant_features = dataset.get('quant_features')
        self.__build_data_set()

    def __build_data_set(self):
        rides = pd.read_csv(self.data_url)
        ## For categorical columns like months and days, we need to generate binary dummy variables
        for field in self.dummy_fields:
            dummies = pd.get_dummies(rides[field], prefix=field, drop_first=False)
            rides = pd.concat([rides, dummies], axis=1)
        data = rides.drop(self.fields_to_drop, axis=1)

        ## Scaling the features to a mean of 0 and standard deviation of 1 for easier network training
        scaled_features = {}
        for feature in self.quant_features:
            mean, std = data[feature].mean(), data[feature].std()
            scaled_features[feature] = [mean, std]
            data.loc[:, feature] = (data[feature] - mean) / std

        test_data = data[-21*24:]
        data = data[:-21*24]

    def return_data_cols(self):
        return self.data
