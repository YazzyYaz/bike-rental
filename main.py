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
        self.__build_data_set()

    def __build_data_set(self):
        rides = pd.read_csv(self.data_url)

    def return_data_cols(self):
        return self.data
