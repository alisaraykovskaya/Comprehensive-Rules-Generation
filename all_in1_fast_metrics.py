from enum import unique
import itertools
import pandas
import fastmetrics
from numba import vectorize, njit
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# import bisect
from itertools import product
from itertools import combinations, permutations
import time
from tqdm import tqdm
from multiprocessing import Queue, Pool, cpu_count, Process, current_process
#from multiprocess import Queue, Pool, cpu_count, Process, current_process
import openpyxl as pxl
import os.path
from multiprocessing.sharedctypes import Value
#from multiprocess.sharedctypes import Value
import boolean
from math import comb
from math import factorial
import re
# from line_profiler_pycharm import profile
from utils import model_string_gen, validate_model, compare_model_similarity, tupleList_to_df, compute_metrics


######################################################################################################################

# Substitute "columns[column_name]" with just "column_name" in formulas
def get_formulas(df):
    df['formula'] = df.apply(lambda x: x['expr'].replace(f'columns[0]', x['columns'][0]), axis=1)
    for i in range(1, len(df['columns'][0])):
        df['formula'] = df.apply(lambda x: x['formula'].replace(f'columns[{i}]', x['columns'][i]), axis=1)
    # df.drop('expr', axis=1, inplace=True)


# Replace boolean python operators with NOT, AND, OR and remove dataframe syntax leftovers
def beautify_formulas(df):
    df['formula'] = df.apply(lambda x: x['formula'].replace('~', 'NOT_'), axis=1)
    df['formula'] = df.apply(lambda x: x['formula'].replace('&', 'AND'), axis=1)
    df['formula'] = df.apply(lambda x: x['formula'].replace('|', 'OR'), axis=1)
    df['formula'] = df.apply(lambda x: x['formula'].replace('df[', ''), axis=1)
    df['formula'] = df.apply(lambda x: x['formula'].replace(']', ''), axis=1)


def add_metrics(best_formulas, y_true):
    for i in range(len(best_formulas)):
        metrics = compute_metrics(y_true, best_formulas[i][-1])
        best_formulas[i] = metrics + best_formulas[i]
    return best_formulas


# @profile
def check_model_perfomance(result, columns, expr, y_true, best_formulas, min_f1):
    f1 = fastmetrics.fast_f1_score(y_true, result)

    if min_f1 == -1 or f1 > min_f1:
        min_f1 = f1
        best_formulas.append((f1, columns, expr, result))

    return best_formulas, min_f1


@njit
def get_result(res, cols, result):
    for i in range(len(result)):
        if not result[i]:
            flag = True

            for j in range(len(res)):
                if res[j] != cols[i][j]:
                    flag = False
                    break

            result[i] = flag


global df
global y_true
global subset_size

global tmp_df
global y_true_tmp
global y_true_np
global df_dict


# @profile
def best_model_helper(expr):
    best_formulas = []
    min_f1 = -1

    bool_pairs = []

    for cnf in expr.replace(" ", "").split('|'):
        bool_pair = []

        for part in cnf.split('&'):
            if part[0] == '~':
                bool_pair.append(False)

            else:
                bool_pair.append(True)

        bool_pairs.append(np.array(bool_pair))

    bool_pairs = np.array(bool_pairs)

    for columns in permutations(df.columns, subset_size):
        result = np.full_like(y_true_np, False)

        df_cols = []
        for col in columns:
            df_cols.append(df_dict[col])

        # .T to switch the rows and cols -->
        # better cache access pattern
        df_cols = np.array(df_cols).T

        for bool_pair in bool_pairs:
            get_result(bool_pair, df_cols, result)

        best_formulas, min_f1 = check_model_perfomance(result, columns, expr, y_true_np, best_formulas, min_f1)

    return best_formulas


# Initalize global varaibles to be passed to each thread
def init_pools(df_temp, y_true_temp, subset_size_temp):
    global df
    global y_true
    global subset_size

    global tmp_df
    global y_true_tmp
    global y_true_np
    global df_dict

    df = df_temp
    y_true = y_true_temp
    subset_size = subset_size_temp

    tmp_df = df[~df.isnull().any(axis=1)]
    y_true_tmp = y_true[~df.isnull().any(axis=1)]
    y_true_np = y_true.values
    df_dict = {}

    for col in df.columns:
        df_dict[col] = df[col].values


def find_best_model(df, y_true, subset_size, parallel=False, num_threads=cpu_count()):
    formulas_number = 2**(2**subset_size)
    best_formulas = []

    start_flag = True
    start = time.time()

    if parallel:
        pool = Pool(processes=num_threads, initializer=init_pools, initargs=(df, y_true, subset_size))
        best_formulas = [pool.map(best_model_helper, tqdm(model_string_gen(subset_size), total=formulas_number-1))]

        # Join together the best models of each thread
        best_formulas = list(itertools.chain.from_iterable(best_formulas))
        best_formulas = list(itertools.chain.from_iterable(best_formulas))

    else:
        init_pools(df, y_true, subset_size)
        for expr in tqdm(model_string_gen(subset_size), total=formulas_number-1):
            best_formulas += best_model_helper(expr)

            if start_flag:
                start_flag = False
                start = time.time()

    elapsed_time = time.time() - start
    print("Elapsed time:", round(elapsed_time, 3), "seconds")
    return best_formulas


def find_best_model_fast_metrics(X_train, y_train, X_test, y_test, subset_size, parallel, num_threads, min_jac_score, file_name, metric):
    best_formulas = find_best_model(X_train, y_train, subset_size, parallel, num_threads)
    best_formulas = sorted(best_formulas, key=lambda tup: tup[0], reverse=True)

    i = 0

    while i < len(best_formulas):
        j = i+1

        while j < len(best_formulas):
            if compare_model_similarity(best_formulas[i][3], best_formulas[j][3], best_formulas[i][1], best_formulas[j][1], metric, min_jac_score):
                del best_formulas[j]
                j -= 1

            j += 1

        i += 1

    best_formulas = add_metrics(best_formulas, y_train)
    models = tupleList_to_df(best_formulas, all_in1=True)
    print('Best model validation results:')
    validate_result = validate_model(models['columns'][0], models['expr'][0], X_test, y_test)
    print(validate_result)
    get_formulas(models)
    beautify_formulas(models)
    models.drop('result', axis=1, inplace=True)
    if os.path.exists(f"./Output/BestModels_{file_name}_all1one.xlsx"):
        with pd.ExcelWriter(f"./Output/BestModels_{file_name}_all1one.xlsx", mode="a", if_sheet_exists='replace', engine="openpyxl") as writer:
            models.to_excel(writer, sheet_name=f'Size {subset_size}', index=False, freeze_panes=(1,1))
    else:
        models.to_excel(f"./Output/BestModels_{file_name}_all1one.xlsx", index=False, freeze_panes=(1,1), sheet_name=f'Size {subset_size}')
    # models.to_excel(f"BestModels_{file_name}.xlsx", index=False, freeze_panes=(1, 1))


######################################################################################################################
