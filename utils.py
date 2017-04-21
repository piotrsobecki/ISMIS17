from os.path import exists
import pandas as pd

class DataFrameCaching():

    def __init__(self,target_file):
        self.target_file = target_file


    def get(self):
        if exists(self.target_file):
            return pd.DataFrame.read()
