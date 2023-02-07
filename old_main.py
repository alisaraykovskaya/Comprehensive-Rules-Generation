from loadData import LoadData, change_column_name
from binarizer import binarize_df
from best_models import find_best_models
from utils import log_exec, get_1var_importance_order, get_1var_importance_config, get_feature_importance, add_missing_features

from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support, cpu_count
import pandas as pd

from os import path, mkdir
import platform
from time import time
from math import factorial
import numpy as np
import random

config = {
    "load_data_params":{
        "project_name": "heart", 
        "load_from_pkl": False
    },

    "rules_generation_params": {
        "quality_metric": "f1", #'f1', 'accuracy', 'rocauc', 'recall', 'precision'
        "subset_size": 3,

        "process_number": 10, # int or "default" = 90% of cpu
        "batch_size": 10000, # number of subsets, which each worker will be processing on every reload
        "filter_similar_between_reloads": False, # If true filter similar models between reloads, otherwise saved in excel best_models between reloads will contain similar models. May lead to not reproducible results.
        
        "crop_number": 1000, # number of best models to compute quality metric threshold
        "crop_number_in_workers": 1000, # same like crop_number, but within the workres. If less than crop_number, it may lead to unstable results
        "excessive_models_num_coef": 3, # how to increase the actual crop_number in order to get an appropriate number of best models after similarity filtering (increase if the result contains too few models)
        
        "dataset_frac": 1,
        "crop_features": 500, # the number of the most important features to remain in a dataset. Needed for reducing working time if dataset has too many features

        "incremental_run": False,
        "crop_features_after_size": 1,
    },
  
    "similarity_filtering_params": {
        "sim_metric": "PARENT", #"JAC_SCORE"
        "min_jac_score": 0.9, #JAC_SCORE threshold
        "min_same_parents": 2 #PARENT sim threshold
    },
  
    "binarizer_params": {
        "unique_threshold": 20, # maximal number of unique values to consider numerical variable as category
        "q": 20, # number of quantiles to split numerical variable, can be lowered if there is need in speed
        "exceptions_threshold": 0.01, # max % of exeptions allowed while converting a variable into numeric to treat it as numeric 
        "numerical_binarization": "range", #"range"
        "nan_threshold": 0.9, # max % of missing values allowed to process the variable
        "share_to_drop": 0.005, # max % of zeros allowed for a binarized column or joint % of ones for the the most unballanced columns which are joined together into 'other' category.
        "create_nan_features": True # If true for every feature that contains nans, corresponding nan indecatore feature will be created
    } 
}

def main():
    if 'windows' in platform.system().lower():
        freeze_support()
    if not path.exists('Output'):
        mkdir('Output')
    if not path.exists('FeatureImportances'):
        mkdir('FeatureImportances')
    if config["rules_generation_params"]["process_number"]=='default':
        config["rules_generation_params"]["process_number"] = int(max(cpu_count()*.9, 1))

    print('LOADING DATA...')
    if config["load_data_params"]["load_from_pkl"] and not path.exists(f'./Data/{config["load_data_params"]["project_name"]}_binarized.pkl'):
        print('Binarized data was not found')
        config["load_data_params"]["load_from_pkl"] = False
    if not config["load_data_params"]["load_from_pkl"]:
        df = LoadData(**config["load_data_params"])
        print('Binarizing data...')
        df, dict_strategies, dict_one_hot_values = binarize_df(df, **config["binarizer_params"])
        df.to_pickle(f'./Data/{config["load_data_params"]["project_name"]}_binarized.pkl')
    else:
        print('Data was loaded from pickle')
        df = pd.read_pickle(f'./Data/{config["load_data_params"]["project_name"]}_binarized.pkl')

    if config['rules_generation_params']['dataset_frac'] != 1:
        df, _ = train_test_split(df, train_size=config['rules_generation_params']['dataset_frac'], stratify=df['Target'], random_state=12)
    
    y_true = df['Target']
    df.drop('Target', axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df, y_true, test_size=0.2, stratify=y_true, random_state=12)
    df_columns = X_train.columns
    columns_number = len(df_columns)
    print(f'project_name: {config["load_data_params"]["project_name"]}  columns_number: {columns_number}  observations_number: {X_train.shape[0]}')

    print('\nRUNNING ON subset_size=1 TO DETERMINE INITIAL FEATURE IMPORTANCES')
    config_1_variable = get_1var_importance_config(config, columns_number)
    start_time = time()
    best_1_variable = find_best_models(X_train, y_train, df_columns, file_name=config_1_variable["load_data_params"]["project_name"], \
                                        **config_1_variable["similarity_filtering_params"],  **config_1_variable["rules_generation_params"])
    elapsed_time = time() - start_time
    log_exec(config_1_variable["load_data_params"]["project_name"], X_train.shape[0], X_train.shape[1], elapsed_time, \
            **config_1_variable["similarity_filtering_params"],  **config_1_variable['rules_generation_params'])
    columns_ordered = get_1var_importance_order(best_1_variable, config["rules_generation_params"]["subset_size"], columns_number)
    if config["rules_generation_params"]["crop_features"] != -1 and (config["rules_generation_params"]["incremental_run"] and config["rules_generation_params"]["crop_features_after_size"] == 1 or not config["rules_generation_params"]["incremental_run"]):
        columns_ordered = columns_ordered[:config["rules_generation_params"]["crop_features"]]
        print(f'\nNumber of features is croped by {config["rules_generation_params"]["crop_features"]}')
    X_train = X_train[columns_ordered]
    print(f'\nTop 5 important features: {columns_ordered[:5]}')

    if not config["rules_generation_params"]["incremental_run"]:
        print('\nBEGIN TRAINING...')
        start_time = time()
        best_models = find_best_models(X_train, y_train, columns_ordered, file_name=config["load_data_params"]["project_name"],\
                        **config["similarity_filtering_params"],  **config['rules_generation_params'])    
        elapsed_time = time() - start_time
        log_exec(config["load_data_params"]["project_name"], X_train.shape[0], X_train.shape[1], elapsed_time, \
                **config["similarity_filtering_params"],  **config['rules_generation_params'])
    else:
        main_subset_size = config["rules_generation_params"]["subset_size"]
        for i in range(2, main_subset_size + 1):
            print(f'\nBEGIN TRAINING FOR subset_size={i}...')
            config["rules_generation_params"]["subset_size"] = i
            start_time = time()
            best_models = find_best_models(X_train, y_train, columns_ordered, file_name=config["load_data_params"]["project_name"],\
                            **config["similarity_filtering_params"],  **config['rules_generation_params'])    
            elapsed_time = time() - start_time
            log_exec(config["load_data_params"]["project_name"], X_train.shape[0], X_train.shape[1], elapsed_time, \
                    **config["similarity_filtering_params"],  **config['rules_generation_params'])
            columns_ordered_new = get_feature_importance(best_models, config["rules_generation_params"]["subset_size"], config["load_data_params"]["project_name"])
            columns_ordered_new = list(columns_ordered_new.index)
            columns_ordered = add_missing_features(columns_ordered, columns_ordered_new)
            if config["rules_generation_params"]["crop_features"] != -1 and config["rules_generation_params"]["crop_features_after_size"] == i:
                columns_ordered = columns_ordered[:config["rules_generation_params"]["crop_features"]]
                print(f'\nNumber of features is croped by {config["rules_generation_params"]["crop_features"]}')
            X_train = X_train[columns_ordered]

if __name__=='__main__':
    main()
