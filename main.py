from loadData import LoadData
from binarizer import binarize_df
from parallel_formula_reload import find_best_model_parallel_formula_reload
from utils import log_exec

from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support, cpu_count
#from multiprocess import freeze_support, cpu_count
import pandas as pd

import os.path
from time import time

# @profile
def main():
    """ Dataset file name settings """
    # file_name = 'COVID_X_train'
    # target_name = 'y'
    # file_ext = 'xlsx'

    # file_name = 'noisy_COVID'
    # target_name = 'y'
    # file_ext = 'csv'

    # file_name = 'heart'
    # target_name = 'HeartDisease'
    # file_ext = 'csv'

    # file_name = 'heart_2020'
    # target_name = 'HeartDisease'
    # file_ext = 'csv'

    # file_name = 'breast-cancer'
    # target_name = 'diagnosis'
    # file_ext = 'csv'

    # file_name = 'water_potability'
    # target_name = 'Potability'
    # file_ext = 'csv'

    # file_name = 'cardio_train'
    # target_name = 'cardio'
    # file_ext = 'csv'

    # file_name = 'hospital_death'
    # target_name = 'hospital_death'
    # file_ext = 'csv'

    # file_name = 'titanic'
    # target_name = 'Survived'
    # file_ext = 'csv'

    # file_name = 'Anonym'
    # target_name = 'Target'
    # file_ext = 'xlsx'

    file_name = 'DivideBy30'
    target_name = 'Div By 30'
    file_ext = 'xlsx'

    # file_name = 'DivideBy30Remainder'
    # target_name = 'Div By 30'
    # file_ext = 'xlsx'

    # file_name = 'Data_Miocarda'
    # target_name = 'Outcome_113_Atrial_fibrillation_'
    # file_ext = 'xlsx'

    """ General settings """
    subset_size = 2
    process_number = 7
    pkl_reload = False

    """
    execution_type options:
        1) parallel_formula_reload"""
    execution_type = 'parallel_formula_reload'

    """
    metrics options: 
    1) JAC_SCORE
    2) PARENT """
    sim_metric = 'PARENT'
    min_jac_score = 0.9
    min_same_parents = 2

    """ parallel_formula_reload settings """
    formula_per_worker = 2
    crop_number = 10000
    crop_number_in_workers = 10000
    filter_similar_between_reloads = False # !!!!!
    workers_filter_similar = False

    """ Binarizer settings """
    unique_threshold=20
    q=20
    exceptions_threshold=0.01
    numerical_binarization='threshold'
    nan_threshold = 0.9
    share_to_drop=0.005 #0.05


    print('Loading data...')
    if not pkl_reload and not os.path.exists(f'./Data/{file_name}_binarized.pkl'):
        print('Binarized data was not found')
        pkl_reload = True
    if pkl_reload:
        df = LoadData(file_name, file_ext, target_name, pkl_reload=pkl_reload)
        print('Binarizing data...')
        df = binarize_df(df, unique_threshold=unique_threshold, q=q, exceptions_threshold=exceptions_threshold,\
            numerical_binarization=numerical_binarization, nan_threshold=nan_threshold, share_to_drop=share_to_drop)
        df.to_pickle(f'./Data/{file_name}_binarized.pkl')
    else:
        print('Data was loaded from pickle')
        df = pd.read_pickle(f'./Data/{file_name}_binarized.pkl')  
    
    y_true = df['Target']
    df.drop('Target', axis=1, inplace=True)
    stratify = y_true
    X_train, X_test, y_train, y_test = train_test_split(df, y_true, test_size=0.2, stratify=stratify, random_state=12)

    if execution_type == 'parallel_formula_reload':
        print('Begin training...')
        start_time = time()
        find_best_model_parallel_formula_reload(X_train, y_train, subset_size=subset_size, metric=sim_metric, min_jac_score=min_jac_score, \
            process_number=process_number, formula_per_worker=formula_per_worker, file_name=file_name, min_same_parents=min_same_parents, \
            filter_similar_between_reloads=filter_similar_between_reloads, crop_number=crop_number, workers_filter_similar=workers_filter_similar, \
            crop_number_in_workers=crop_number_in_workers)
        elapsed_time = time() - start_time
        log_exec(file_name, execution_type, sim_metric, subset_size, X_train.shape[0], X_train.shape[1], elapsed_time, process_number, formula_per_worker)

if __name__=='__main__':
    # Windows flag
    freeze_support()
    main()
