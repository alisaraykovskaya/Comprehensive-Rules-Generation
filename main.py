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
import json
import copy

def main():
    if 'windows' in platform.system().lower():
        freeze_support()
    if not path.exists('Output'):
        mkdir('Output')
    with open('./config.json') as config_file:
        config = json.load(config_file) 
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

    if config['rules_generation_params']['subset_size'] != 1:
        print('DETERMINING FEATURE IMPORTANCES...')
        config_1_variable = copy.deepcopy(config)
        config_1_variable['rules_generation_params']['subset_size'] = 1
        config_1_variable['rules_generation_params']['process_number'] = 2
        config_1_variable['rules_generation_params']['formula_per_worker'] = 1
        best_1_variable = find_best_model_parallel_formula_reload(X_train, y_train, file_name = config_1_variable["load_data_params"]["file_name"], **config_1_variable["similarity_filtering_params"],  **config_1_variable["rules_generation_params"])
        columns_ordered = get_importance_order(best_1_variable)
        if config["rules_generation_params"]["crop_features"] != -1:
            columns_ordered = columns_ordered[:config["rules_generation_params"]["crop_features"]]
            print(f'Top {config["rules_generation_params"]["crop_features"]} important features: {columns_ordered}')
        X_train = X_train[columns_ordered]

    print('BEGIN TRAINING...')
    start_time = time()
    find_best_model_parallel_formula_reload(X_train, y_train, file_name = config["load_data_params"]["file_name"], **config["similarity_filtering_params"],  **config["rules_generation_params"])
    
    elapsed_time = time() - start_time
    log_exec(config["load_data_params"]["file_name"], config["similarity_filtering_params"]["sim_metric"], \
            config["rules_generation_params"]["subset_size"], X_train.shape[0], X_train.shape[1], \
            elapsed_time, config["rules_generation_params"]["process_number"], config["rules_generation_params"]["formula_per_worker"])

if __name__=='__main__':
    main()
