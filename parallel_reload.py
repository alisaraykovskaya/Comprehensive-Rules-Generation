import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import fastmetrics
from numba import njit
from tqdm import tqdm

import os.path
from enum import unique
# import bisect
from itertools import product, combinations, permutations, chain
import time
from math import comb
from math import factorial
import re

from multiprocessing import Queue, Pool, cpu_count, Process, current_process
#from multiprocess import Queue, Pool, cpu_count, Process, current_process
from multiprocessing.sharedctypes import Value
#from multiprocess.sharedctypes import Value
import openpyxl as pxl
import boolean

from utils import model_string_gen, count_confusion_matrix, simplify_expr, add_readable_simple_formulas, find_sum, sums_generator
from utils import tupleList_to_df, beautify_simple, beautify_summed, compute_complexity_metrics, get_parent_set, compare_model_similarity


'''Reload parallel execution'''#############################################################################


def worker_init(df_tmp, y_true_tmp, best_models_main_tmp, subset_size_tmp, metric_tmp, min_jac_score_tmp, min_f1_tmp, start_time_tmp, \
        workers_filter_similar_tmp):
    global df
    global y_true
    global subset_size
    global metric
    global min_jac_score
    global best_models
    global best_models_main
    global min_f1
    global start_time
    global workers_filter_similar

    global y_true_np
    global df_dict

    df = df_tmp
    y_true = y_true_tmp
    subset_size = subset_size_tmp
    metric = metric_tmp
    min_jac_score = min_jac_score_tmp
    min_f1 = min_f1_tmp
    start_time = start_time_tmp
    best_models_main = best_models_main_tmp
    workers_filter_similar = workers_filter_similar_tmp
    best_models = []

    df['target'] = y_true
    df.dropna(inplace=True)
    y_true_np = df['target'].values
    df.drop(columns=['target'], inplace=True)
    df_dict = {}
    for col in df.columns:
        df_dict[col] = df[col].values


global df
global y_true
global subset_size
global metric
global min_jac_score
global best_models
global best_models_main
global min_f1
global start_time
global workers_filter_similar

global y_true_np
global df_dict


def worker_reload(formula_template, expr, summed_expr, columns):
    # model = eval('lambda df, columns: ' + expr)

    columns = list(columns)

    # tmp_df = df[columns]
    # tmp_idx = ~tmp_df.isnull().any(axis=1)
    # tmp_df = tmp_df[tmp_idx]
    # y_true_tmp = y_true[tmp_idx].values

    df_np_cols = []
    for col in columns:
        # df_np_cols.append(df_dict[col][tmp_idx])
        df_np_cols.append(df_dict[col])
    df_np_cols = np.array(df_np_cols)

    # result = model(tmp_df, columns).to_numpy()
    result = formula_template(df_np_cols)

    # tp, fp, fn, tn = count_confusion_matrix(y_true_tmp, result)
    tp, fp, fn, tn = count_confusion_matrix(y_true_np, result)
    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
    # Check if model passes theshold
    f1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
    if f1 > min_f1:
        fpr = 0 if (fp + tn) == 0 else fp / (fp + tn)
        rocauc = (1 + recall - fpr) / 2
        accuracy = 0 if (tp + fp + fn + tn) == 0 else (tp + tn) / (tp + fp + fn + tn)
        precision_0 = 0 if (tn + fn) == 0 else tn / (tn + fn)
        recall_0 = 0 if (tn + fp) == 0 else tn / (tn + fp)
        f1_0 = 0 if (precision_0 + recall_0) == 0 else 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
        columns_set = get_parent_set(columns)
        elapsed_time = time.time() - start_time
        model_info = {'f1_1': f1, 'f1_0': f1_0, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'precision_1': precision, 'precision_0': precision_0,\
            'recall_1': recall, 'recall_0': recall_0, 'rocauc': rocauc, 'accuracy': accuracy, 'elapsed_time': elapsed_time, \
            'summed_expr': summed_expr, 'columns': columns, 'expr': expr, 'columns_set': columns_set, 'result': result}
        return model_info
    else:
        return None


def find_best_model_parallel_reload(df, y_true, subset_size, metric, min_jac_score, batch_ratio=1, process_number=False, \
        file_name='tmp', filter_similar_between_reloads=False, crop_number=None, workers_filter_similar=False):
    excel_exist = False
    if os.path.exists(f"./Output/BestModels_{file_name}_reload.xlsx"):
        excel_exist = True

    df_columns = df.columns
    columns_number = len(df_columns)
    formulas_number = 2**(2**subset_size) - 2
    subset_number = factorial(columns_number) / factorial(columns_number - subset_size)
    total_count = formulas_number * subset_number
    print(f'columns_number={columns_number}, formulas_number={formulas_number}, subset_number={subset_number}, total_count={total_count}')
    batch_size = int(total_count * batch_ratio)
    print(batch_size)

    # variables (one-charactered) and algebra are needed for simplifying expressions
    variables = list(map(chr, range(122, 122-subset_size,-1)))
    algebra = boolean.BooleanAlgebra()

    # set of simplified expressions, because no need of processing same expressions
    expr_set = set()

    if not process_number:
        process_number = cpu_count()-5
    start_time = time.time()

    best_models = []
    min_f1 = -1
    finish = False
    models_batch = []
    overall_count = 0
    current_count = 0

    for expr in tqdm(model_string_gen(subset_size), total=formulas_number):        
        simple_expr = simplify_expr(expr, subset_size, variables, algebra)

        # sums = sums_generator(subset_size)
        summed_expr = find_sum(sums_generator(subset_size), simple_expr)

        # Replace one-charachter variables in simplified expr with 'df[columns[{i}]]'
        # for easy execution
        for i in range(subset_size):
            simple_expr = simple_expr.replace(variables[i], f'df[columns[{i}]]')

        # If formula is a tautology
        if simple_expr == '1':
            continue

        # Check if expr already processed
        # if simple_expr not in expr_set:
        #     expr_set.add(simple_expr)
        # else:
        #     total_count -= subset_number
        #     continue

        simple_tmp = simple_expr.replace('df', 'df_np_cols')
        for i in range(subset_size):
            simple_tmp = simple_tmp.replace(f'columns[{i}]', f'{i}')
        formula_template = eval(f'njit(lambda df_np_cols: {simple_tmp})')

        # Columns batch creation and giving to workers
        for columns in permutations(df_columns, subset_size):
            models_batch.append((formula_template, simple_expr, summed_expr, columns))
            current_count += 1
            overall_count += 1
            if current_count >= batch_size:
                pool = Pool(process_number-1, initializer=worker_init, initargs=(df, y_true, best_models, subset_size, metric, \
                            min_jac_score, min_f1, start_time, workers_filter_similar))
                new_models = pool.starmap(worker_reload, models_batch)
                pool.close()
                pool.join()
                print('\nGot models')
                best_models = list(chain.from_iterable([best_models, new_models]))
                best_models = list(filter(None, best_models))
                best_models = sorted(best_models, key=lambda row: row['f1_1'], reverse=True)
                # if filter_similar_between_reloads:
                #     similarity_filtering()
                if crop_number is not None:
                    best_models = best_models[:crop_number+1]
                min_f1 = best_models[-1]['f1_1']
                print(min_f1)
                best_models = add_readable_simple_formulas(best_models, subset_size)

                # Preparing models to be written in excel
                models_to_excel = tupleList_to_df(best_models, reload=True)
                models_to_excel.drop(['columns_set', 'result'], axis=1, inplace=True)
                beautify_simple(models_to_excel)
                beautify_summed(models_to_excel, subset_size, variables)
                compute_complexity_metrics(models_to_excel)

                if excel_exist:
                    with pd.ExcelWriter(f"./Output/BestModels_{file_name}_reload.xlsx", mode="a", if_sheet_exists='replace', engine="openpyxl") as writer:
                        models_to_excel.to_excel(writer, sheet_name=f'Size {subset_size}', index=False, freeze_panes=(1,1))
                else:
                    models_to_excel.to_excel(f"./Output/BestModels_{file_name}_reload.xlsx", index=False, freeze_panes=(1,1), sheet_name=f'Size {subset_size}')
                    excel_exist = True
                models_batch = []
                current_count = 0
                print(f'processed {(overall_count/total_count) * 100:.1f}% models')
                if overall_count >= total_count:
                    finish = True
    if not finish:
        pool = Pool(process_number-1, initializer=worker_init, initargs=(df, y_true, best_models, subset_size, metric, \
                    min_jac_score, min_f1, start_time, workers_filter_similar))
        # print('AAAAAAAAAAAAAAAA')
        new_models = pool.starmap(worker_reload, models_batch)
        pool.close()
        pool.join()
        print('Got models')
        best_models = list(chain.from_iterable([best_models, new_models]))
        best_models = list(filter(None, best_models))
        best_models = sorted(best_models, key=lambda row: row['f1_1'], reverse=True)
        best_models = add_readable_simple_formulas(best_models, subset_size)
        # if filter_similar_between_reloads:
        #     similarity_filtering()
        # if crop_number is not None:
        #     best_models = best_models[:crop_number+1]
        min_f1 = best_models[-1]['f1_1']

        # Preparing models to be written in excel
        models_to_excel = tupleList_to_df(best_models, reload=True)
        models_to_excel.drop(['columns_set', 'result'], axis=1, inplace=True)
        beautify_simple(models_to_excel)
        beautify_summed(models_to_excel, subset_size, variables)
        compute_complexity_metrics(models_to_excel)

        if excel_exist:
            with pd.ExcelWriter(f"./Output/BestModels_{file_name}_reload.xlsx", mode="a", if_sheet_exists='replace', engine="openpyxl") as writer:
                models_to_excel.to_excel(writer, sheet_name=f'Size {subset_size}', index=False, freeze_panes=(1,1))
        else:
            models_to_excel.to_excel(f"./Output/BestModels_{file_name}_reload.xlsx", index=False, freeze_panes=(1,1), sheet_name=f'Size {subset_size}')
            excel_exist = True
            models_batch = []
            current_count = 0
        print(f'processed {(overall_count/total_count) * 100:.1f}% models')
    return


######################################################################################################################