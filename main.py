from loadData import LoadData
from binarizer import binarize_df
from best_models import find_best_model_shared_f1_batch_parallel
from sequential_exec import find_best_model_sequential
from utils import log_exec

from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support, cpu_count
#from multiprocess import freeze_support, cpu_count
import pandas as pd

import os.path
from time import time


if __name__=='__main__':
    # Windows flag
    freeze_support()

    """ Dataset file name settings """
    # file_name = 'COVID_X_train'
    # target_name = 'y'
    # file_ext = 'xlsx'

    # file_name = 'noisy_COVID'
    # target_name = 'y'
    # file_ext = 'csv'

    file_name = 'heart'
    target_name = 'HeartDisease'
    file_ext = 'csv'

    # file_name = 'heart_2020'
    # target_name = 'HeartDisease'
    # file_ext = 'csv'

    # file_name = 'breast-cancer'
    # target_name = 'diagnosis'
    # file_ext = 'csv'

    # file_name = 'water_potability'
    # target_name = 'Potability'
    # file_ext = 'csv'

    #file_name = 'cardio_train'
    #target_name = 'cardio'
    #file_ext = 'csv'

    #file_name = 'hospital_death'
    #target_name = 'hospital_death'
    #file_ext = 'csv'
    
    #file_name = 'titanic'
    #target_name = 'Survived'
    #file_ext = 'csv'

    # file_name = 'Anonym'
    # target_name = 'Target'
    # file_ext = 'xlsx'
    
    # file_name = 'Data_Miocarda'
    # target_name = 'Outcome_113_Atrial_fibrillation_'
    # file_ext = 'xlsx'

    """ Changeable model settings """
    subset_size = 3
    process_number = cpu_count() - 3
    #process_number = 13
    pkl_reload = False
    execution_type = 'parallel_orig'
    # types: sequential parallel_orig parallel_reload 

    """ Binarizer settings"""
    unique_threshold=20
    q=5
    exceptions_threshold=0.01
    numerical_binarization='range'
    nan_threshold = 0.9
    share_to_drop=0.005 #0.05

    """ Better not change model settings"""
    batch_size = 500
    parallel = True
    
    print('Loading data...')
    if not pkl_reload and not os.path.exists(f'./Data/{file_name}_binarized.pkl'):
        print('Binarized data was not found')
        pkl_reload = True
    if pkl_reload:
        df = LoadData(file_name, file_ext, target_name, pkl_reload=pkl_reload)
        print('Binarizing data...')
        df = binarize_df(df, unique_threshold=unique_threshold, q=q, exceptions_threshold=exceptions_threshold, numerical_binarization=numerical_binarization, nan_threshold=nan_threshold, share_to_drop=share_to_drop)
        df.to_pickle(f'./Data/{file_name}_binarized.pkl')
    else:
        print('Data was loaded from pickle')
        df = pd.read_pickle(f'./Data/{file_name}_binarized.pkl')
        
    
    y_true = df['Target']
    df.drop('Target', axis=1, inplace=True)
    stratify = y_true
    X_train, X_test, y_train, y_test = train_test_split(df, y_true, test_size=0.3, stratify=stratify, random_state=12)
    


    """ NOT PARALLEL TRAINING """
    if not parallel:
        print('Begin training...')
        start_time = time()
        find_best_model_sequential(X_train, y_train, X_test, y_test, subset_size=subset_size, file_name=file_name)
        elapsed_time = time() - start_time
        log_exec(execution_type, subset_size, elapsed_time, process_number, batch_size)

    """ PARALLEL TRAINING """
    if parallel:
        print('Begin training...')
        start_time = time()
        find_best_model_shared_f1_batch_parallel(X_train, y_train, subset_size=subset_size, process_number=process_number, batch_size=batch_size, file_name=file_name)
        elapsed_time = time() - start_time
        log_exec(execution_type, subset_size, elapsed_time, process_number, batch_size)