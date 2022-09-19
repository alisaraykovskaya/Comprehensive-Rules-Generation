from loadData import LoadData
from binarizer import binarize_df
from parallel_formula_reload import find_best_model_parallel_formula_reload
from utils import log_exec

from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support, cpu_count
import pandas as pd

import os.path
from time import time
import json

def main():
    with open('./config.json') as config_file:
        config = json.load(config_file) 
    if config["rules_generation_params"]["process_number"]=='default':
        config["rules_generation_params"]["process_number"] = int(cpu_count()*0.9)

    print('Loading data...')
    if not config["load_data_params"]["pkl_reload"] and not os.path.exists(f'./Data/{config["load_data_params"]["file_name"]}_binarized.pkl'):
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

    print('Begin training...')
    start_time = time()
    find_best_model_parallel_formula_reload(X_train, y_train, file_name = config["load_data_params"]["file_name"], **config["similarity_filtering_params"],  **config["rules_generation_params"])
    
    elapsed_time = time() - start_time
    log_exec(config["load_data_params"]["file_name"], config["similarity_filtering_params"]["sim_metric"], \
            config["rules_generation_params"]["subset_size"], X_train.shape[0], X_train.shape[1], \
            elapsed_time, config["rules_generation_params"]["process_number"], config["rules_generation_params"]["formula_per_worker"])

if __name__=='__main__':
    # Windows flag
    freeze_support()
    main()
