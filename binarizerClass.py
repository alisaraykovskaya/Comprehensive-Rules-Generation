import pandas as pd
from os import path

from binarizer import binarize, boolarize, binarizer_predict


class Binarizer:
    def __init__(
        self,
        project_name='',
        load_from_pkl=True,
        unique_threshold=20,
        q=20,
        exceptions_threshold=0.01,
        numerical_binarization='threshold',
        nan_threshold=0.8,
        share_to_drop=0.05,
        create_nan_features=True
    ):
        self.project_name = project_name
        self.load_from_pkl = load_from_pkl
        self.unique_threshold = unique_threshold
        self.q = q
        self.exceptions_threshold = exceptions_threshold
        self.numerical_binarization = numerical_binarization
        self.nan_threshold = nan_threshold
        self.share_to_drop = share_to_drop
        self.create_nan_features = create_nan_features

        self.dict_strategies = {}
        self.dict_one_hot_values = {}


    def fit_transform(self, df):
        if self.load_from_pkl and not path.exists(f'./Data/{self.project_name}_train_binarized.pkl'):
            print('Binarized train data was not found')
            self.load_from_pkl = False
        
        if self.load_from_pkl:
            print('Train data was loaded from pickle')
            df = pd.read_pickle(f'./Data/{self.project_name}_train_binarized.pkl')
            return df

        dict_strategies = {}
        dict_one_hot_values = {}

        for column in df.columns:
            if column != 'Target':
                print(column)
                binarized, dict_strategies, dict_one_hot_values = binarize(df, column, self.unique_threshold, self.q, self.exceptions_threshold, self.numerical_binarization, self.nan_threshold, self.share_to_drop, self.create_nan_features, dict_strategies, dict_one_hot_values)
                if binarized is not None:
                    df = df.join(binarized)
                    df.drop(column, axis=1, inplace=True)
                else:
                    df.drop(column, axis=1, inplace=True)
        
        # convert to boolean type
        for col in df.columns:
            df[col] = df[col].apply(boolarize)

        self.dict_strategies = dict_strategies
        self.dict_one_hot_values = dict_one_hot_values

        if not self.load_from_pkl:
            df.to_pickle(f'./Data/{self.project_name}_train_binarized.pkl')

        return df

    
    def transform(self, df):
        if self.load_from_pkl and not path.exists(f'./Data/{self.project_name}_test_binarized.pkl'):
            print('Binarized test data was not found')
            self.load_from_pkl = False
        
        if self.load_from_pkl:
            print('Test data was loaded from pickle')
            df = pd.read_pickle(f'./Data/{self.project_name}_test_binarized.pkl')
            return df

        for column in df.columns:
            if column != 'Target' and column in self.dict_strategies.keys():
                binarized_ = binarizer_predict(df, column, self.dict_strategies, self.dict_one_hot_values)
                df = df.join(binarized_)
                df.drop(column, axis=1, inplace=True)
            elif column == 'Target':
                df.loc[(df['Target']==1), 'Target'] = True
                df.loc[(df['Target']==0), 'Target'] = False
            else:
                df.drop(column, axis=1, inplace=True)

        if not self.load_from_pkl:
            df.to_pickle(f'./Data/{self.project_name}_test_binarized.pkl')
        return df
