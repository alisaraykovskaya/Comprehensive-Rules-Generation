from loadData import LoadData
from binarizer import binarize_df
from parallel_batch_continuous import find_best_model_shared_f1_batch_parallel
from sequential_exec import find_best_model_sequential
from all_in1_fast_metrics import find_best_model_fast_metrics
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
    target_name = 'Divisible by 30'
    file_ext = 'csv'

    # file_name = 'Data_Miocarda'
    # target_name = 'Outcome_113_Atrial_fibrillation_'
    # file_ext = 'xlsx'

    """ Changeable model settings """
    subset_size = 3
    process_number = cpu_count() - 3
    #process_number = 13
    pkl_reload = False
    min_jac_score = .90
    execution_type = 'all_in1_fast_metrics_parallel'
    # types: all_in1_fast_metrics_parallel | parallel_batch_continuous | all_in1_fast_metrics_seq | sequential
    sim_metric = 'JAC_SCORE'
    """ metrics: 
    JAC_SCORE - compatible with everything
    PARENT(list) - for all_in1 versions
    PARENT(set) - for batch_continuous versions
    """


    """ Binarizer settings"""
    unique_threshold=20
    q=5
    exceptions_threshold=0.01
    numerical_binarization='range'
    nan_threshold = 0.9
    share_to_drop=0.005 #0.05

    """ Better not change model settings"""
    batch_size = 500

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
    X_train, X_test, y_train, y_test = train_test_split(df, y_true, test_size=0.3, stratify=stratify, random_state=12)

    if execution_type == 'sequential':
        print('Begin training...')
        start_time = time()
        find_best_model_sequential(X_train, y_train, X_test, y_test, subset_size=subset_size, file_name=file_name)
        elapsed_time = time() - start_time
        log_exec(file_name, execution_type, sim_metric, subset_size, elapsed_time, process_number, batch_size)

    elif execution_type == 'parallel_batch_continuous':
        print('Begin training...')
        start_time = time()
        find_best_model_shared_f1_batch_parallel(X_train, y_train, subset_size=subset_size, metric=sim_metric, min_jac_score=min_jac_score, \
            process_number=process_number, batch_size=batch_size, file_name=file_name)
        elapsed_time = time() - start_time
        log_exec(file_name, execution_type, '-', subset_size, elapsed_time, process_number, batch_size)

    elif execution_type == 'all_in1_fast_metrics_parallel':
        print('Begin training...')
        start_time = time()
        find_best_model_fast_metrics(X_train, y_train, X_test, y_test, subset_size=subset_size, parallel=True, \
            num_threads=process_number, min_jac_score=min_jac_score, file_name=file_name, metric=sim_metric)
        elapsed_time = time() - start_time
        log_exec(file_name, execution_type, sim_metric, subset_size, elapsed_time, process_number, '-')

    elif execution_type == 'all_in1_fast_metrics_seq':
        print('Begin training...')
        start_time = time()
        find_best_model_fast_metrics(X_train, y_train, X_test, y_test, subset_size=subset_size, parallel=False, \
            num_threads=process_number, min_jac_score=min_jac_score, file_name=file_name, metric=sim_metric)
        elapsed_time = time() - start_time
        log_exec(file_name, execution_type, sim_metric, subset_size, elapsed_time, process_number, '-')

if __name__=='__main__':
    # Windows flag
    freeze_support()
    main()
