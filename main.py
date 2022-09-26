from loadData import LoadData
from binarizer import binarize_df
from parallel_formula_reload import find_best_model_parallel_formula_reload
from utils import log_exec, get_importance_order

from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support, cpu_count
import pandas as pd

from os import path, mkdir
import platform
from time import time
from math import factorial
import json
import copy

config = {
    "load_data_params2":{
        "file_name": "DivideBy30",
        "target_name": "Div By 30",
        "file_ext": "xlsx", 
        "pkl_reload": False
    },
    "load_data_params":{
        "file_name": "heart",
        "target_name": "cardio",
        "file_ext": "csv", 
        "pkl_reload": False
    },

    "rules_generation_params": {
        "quality_metric": "f1",
        "subset_size": 2,
        "process_number": 2,
        "formula_per_worker": 1,
        "crop_features": -1,
        "crop_number": 10000,
        "excessive_models_num_coef": 3,
        "crop_number_in_workers": 10000,
        "dropna_on_whole_df": False,
        "desired_minutes_per_worker": 10
    },
  
    "similarity_filtering_params": {
        "sim_metric": "PARENT",
        "min_jac_score": 0.9,
        "min_same_parents": 2
    },
  
    "binarizer_params": {
        "unique_threshold": 20,
        "q": 20,
        "exceptions_threshold": 0.01,
        "numerical_binarizatio": "threshold",
        "nan_threshold": 0.9,
        "share_to_drop": 0.005
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
    if not config["load_data_params"]["pkl_reload"] and not path.exists(f'./Data/{config["load_data_params"]["file_name"]}_binarized.pkl'):
        print('Binarized data was not found')
        config["load_data_params"]["pkl_reload"] = True
    if config["load_data_params"]["pkl_reload"]:
        df = LoadData(**config["load_data_params"])
        print('Binarizing data...')
        df = binarize_df(df, **config["binarizer_params"])
        df.to_pickle(f'./Data/{config["load_data_params"]["file_name"]}_binarized.pkl')
    else:
        print('Data was loaded from pickle')
        df = pd.read_pickle(f'./Data/{config["load_data_params"]["file_name"]}_binarized.pkl')  
    
    y_true = df['Target']
    df.drop('Target', axis=1, inplace=True)
    stratify = y_true
    X_train, X_test, y_train, y_test = train_test_split(df, y_true, test_size=0.2, stratify=stratify, random_state=12)
    df_columns = X_train.columns
    columns_number = len(df_columns)
    formulas_number = 2**(2**config['rules_generation_params']['subset_size']) - 2
    models_per_formula = factorial(columns_number) / factorial(columns_number - config['rules_generation_params']['subset_size'])
    total_count = formulas_number * models_per_formula
    print(f'df_name: {config["load_data_params"]["file_name"]}  columns_number: {columns_number}  observations_number: {X_train.shape[1]}  columns_number: {columns_number}  formulas_number: {formulas_number}  models_per_formula: {models_per_formula}  total_count_of_models: {total_count}')

    if config['rules_generation_params']['subset_size'] != 1:
        print('\nDETERMINING FEATURE IMPORTANCES...')
        config_1_variable = copy.deepcopy(config)
        config_1_variable['rules_generation_params']['subset_size'] = 1
        config_1_variable['rules_generation_params']['process_number'] = 2
        config_1_variable['rules_generation_params']['formula_per_worker'] = 1
        best_1_variable, average_time_per_model = find_best_model_parallel_formula_reload(X_train, y_train, file_name = config_1_variable["load_data_params"]["file_name"], **config_1_variable["similarity_filtering_params"],  **config_1_variable["rules_generation_params"])
        columns_ordered = get_importance_order(best_1_variable)
        if config["rules_generation_params"]["crop_features"] != -1:
            columns_ordered = columns_ordered[:config["rules_generation_params"]["crop_features"]]
            print(f'Top {config["rules_generation_params"]["crop_features"]} important features: {columns_ordered}')
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
    find_best_model_parallel_formula_reload(X_train, y_train, file_name = config["load_data_params"]["file_name"], **config["similarity_filtering_params"],  **config["rules_generation_params"])
    
    elapsed_time = time() - start_time
    log_exec(config["load_data_params"]["file_name"], config["similarity_filtering_params"]["sim_metric"], \
            config["rules_generation_params"]["subset_size"], X_train.shape[0], X_train.shape[1], \
            elapsed_time, config["rules_generation_params"]["process_number"], config["rules_generation_params"]["formula_per_worker"])

if __name__=='__main__':
    main()
