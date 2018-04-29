import pandas as pd
from config import dataset
import pprint

class Dataset(object):
    """Dataset Class to instantiate test and training datasets for bike"""
    def __init__(self, data_url):
        self.data_url = data_url
        self.dummy_variables = dataset.get('dummy_variables')
        self.fields_to_drop = dataset.get('fields_to_drop')
        self.quant_features = dataset.get('quant_features')
        self.target_fields = dataset.get('target_fields')
        self.__build_data_set()

    def __build_data_set(self):
        self.rides = pd.read_csv(self.data_url)
        ## For categorical columns like months and days, we need to generate binary dummy variables
        for field in self.dummy_variables:
            dummies = pd.get_dummies(self.rides[field], prefix=field, drop_first=False)
            self.rides = pd.concat([self.rides, dummies], axis=1)
        data = self.rides.drop(self.fields_to_drop, axis=1)

        ## Scaling the features to a mean of 0 and standard deviation of 1 for easier network training
        self.scaled_features = {}
        for feature in self.quant_features:
            mean, std = data[feature].mean(), data[feature].std()
            self.scaled_features[feature] = [mean, std]
            data.loc[:, feature] = (data[feature] - mean) / std

        ## Split the data in test, training and validation data (validation happens on last 60 days of train data)
        self.test_data = data[-21*24:]
        data = data[:-21*24]
        features, targets = data.drop(self.target_fields, axis=1), data[self.target_fields]

        self.test_features, self.test_targets = self.test_data.drop(self.target_fields, axis=1), self.test_data[self.target_fields]
        self.train_features, self.train_targets = features[:-60*24], targets[:-60*24]
        self.val_features, self.val_targets = features[-60*24:], targets[-60*24:]

    def return_test_data(self):
        return self.test_features, self.test_targets

    def return_train_data(self):
        return self.train_features, self.train_targets

    def return_val_features(self):
        return self.val_features, self.val_targets

    def return_scaled_feature(self):
        return self.scaled_features

    def return_rides(self):
        return self.rides
