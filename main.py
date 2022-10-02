from loadData import LoadData
from binarizer import binarize_df
from parallel_formula_reload import find_best_model_parallel_formula_reload
from utils import log_exec, get_importance_order, create_feature_importance_config

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
        "project_name": "DivideBy30RemainderNull", 
        "pkl_reload": False
    },

    "rules_generation_params": {
        "quality_metric": "f1", #'f1', 'accuracy', 'rocauc', 'recall', 'precision'
        "subset_size": 1,
        "process_number": 2, # int or "defalut" = 90% of cpu
        "formula_per_worker": 1, # number of formulas passed to each worker in each batch
        "crop_features": -1, # the number of the most important features to remain in a dataset. Needed for reducing working time if dataset has too many features
        "crop_number": 1000, # number of best models to compute quality metric threshold
        "crop_number_in_workers": 1000, # same like crop_number, but within the workres. If less than crop_number, it may lead to unstable results
        "excessive_models_num_coef": 1.2, # how to increase the actual crop_number in order to get an appropriate number of best models after similarity filtering (increase if the result contains too few models)
        "dropna_on_whole_df": True, #If True, then dropna will be performed on whole dataset before algorithm, otherwise dropna will be used for every model individually to it's subset of columns (which is slow).
        "desired_minutes_per_worker": 20,
        "filter_similar_between_reloads": False # If true filter similar models between reloads, otherwise saved in excel best_models between reloads will contain similar models. May lead to not reproducible results.
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
        "numerical_binarization": "threshold", #"range"
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
    if config["rules_generation_params"]["process_number"]=='default':
        config["rules_generation_params"]["process_number"] = int(max(cpu_count()*.9, 1))

    print('LOADING DATA...')
    if not config["load_data_params"]["pkl_reload"] and not path.exists(f'./Data/{config["load_data_params"]["project_name"]}_binarized.pkl'):
        print('Binarized data was not found')
        config["load_data_params"]["pkl_reload"] = True
    if config["load_data_params"]["pkl_reload"]:
        df = LoadData(**config["load_data_params"])
        print('Binarizing data...')
        df = binarize_df(df, **config["binarizer_params"])
        df.to_pickle(f'./Data/{config["load_data_params"]["project_name"]}_binarized.pkl')
    else:
        print('Data was loaded from pickle')
        df = pd.read_pickle(f'./Data/{config["load_data_params"]["project_name"]}_binarized.pkl')
    
    y_true = df['Target']
    df.drop('Target', axis=1, inplace=True)
    stratify = y_true
    X_train, X_test, y_train, y_test = train_test_split(df, y_true, test_size=0.2, stratify=stratify, random_state=12)
    df_columns = X_train.columns
    columns_number = len(df_columns)
    formulas_number = 2**(2**config['rules_generation_params']['subset_size']) - 2
    models_per_formula = factorial(columns_number) / factorial(columns_number - config['rules_generation_params']['subset_size'])
    total_count = formulas_number * models_per_formula
    print(f'project_name: {config["load_data_params"]["project_name"]}  columns_number: {columns_number}  observations_number: {X_train.shape[0]}  columns_number: {columns_number}  formulas_number: {formulas_number}  models_per_formula: {models_per_formula}  total_count_of_models: {total_count}')

    if config['rules_generation_params']['subset_size'] != 1:
        print('\nDETERMINING FEATURE IMPORTANCES...')
        config_1_variable = create_feature_importance_config(config, columns_number)
        best_1_variable, average_time_per_model = find_best_model_parallel_formula_reload(X_train, y_train, file_name = config_1_variable["load_data_params"]["project_name"], **config_1_variable["similarity_filtering_params"],  **config_1_variable["rules_generation_params"])
        columns_ordered = get_importance_order(best_1_variable)
        if config["rules_generation_params"]["crop_features"] != -1:
            columns_ordered = columns_ordered[:config["rules_generation_params"]["crop_features"]]
        X_train = X_train[columns_ordered]
        models_per_formula = factorial(columns_number) / factorial(columns_number - config['rules_generation_params']['subset_size'])
        minutes_per_formula = average_time_per_model * models_per_formula / 60
        desired_minutes_per_worker = config['rules_generation_params']['desired_minutes_per_worker']
        if minutes_per_formula > desired_minutes_per_worker:
            print(f'One formula takes time more than desired time per worker ({minutes_per_formula}), could not determine needed settings')
        else:
            print(f'Approximate number of formulas per process for each process to work {desired_minutes_per_worker} minutes on subset_size={config["rules_generation_params"]["subset_size"]}: {round(desired_minutes_per_worker / minutes_per_formula)}')
        print(f'Top 5 important features: {columns_ordered[:5]}')


    print('\nBEGIN TRAINING...')
    start_time = time()
    find_best_model_parallel_formula_reload(X_train, y_train, file_name = config["load_data_params"]["project_name"], **config["similarity_filtering_params"],  **config["rules_generation_params"])
    
    elapsed_time = time() - start_time
    log_exec(config["load_data_params"]["project_name"], config["similarity_filtering_params"]["sim_metric"], \
            config["rules_generation_params"]["subset_size"], X_train.shape[0], X_train.shape[1], \
            elapsed_time, config["rules_generation_params"]["process_number"], config["rules_generation_params"]["formula_per_worker"])

if __name__=='__main__':
    main()
