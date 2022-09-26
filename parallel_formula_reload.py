import pandas as pd
import numpy as np
from numba import njit

import os.path
from enum import unique
from itertools import product, combinations, permutations, chain
import time
from math import factorial

from multiprocessing import Queue, Pool, cpu_count, Process, current_process
#from multiprocess import Queue, Pool, cpu_count, Process, current_process
from multiprocessing.sharedctypes import Value
#from multiprocess.sharedctypes import Value
import boolean

from utils import model_string_gen, simplify_expr, add_readable_simple_formulas, find_sum, sums_generator
from utils import tupleList_to_df, beautify_simple, beautify_summed, similarity_filtering
from metrics_utils import count_confusion_matrix, compute_complexity_metrics, get_parent_set


'''Reload parallel execution'''#############################################################################


def worker_init(df_tmp, y_true_tmp, subset_size_tmp, quality_metric_tmp, sim_metric_tmp, min_jac_score_tmp, min_same_parents_tmp, min_f1_tmp, start_time_tmp, \
        workers_filter_similar_tmp, crop_number_in_workers_tmp):
    global df
    global y_true
    global subset_size
    global quality_metric
    global sim_metric
    global min_jac_score
    global min_same_parents
    global min_f1
    global start_time
    global workers_filter_similar
    global crop_number_in_workers

    global y_true_np
    global df_dict

    df = df_tmp
    y_true = y_true_tmp
    subset_size = subset_size_tmp
    quality_metric = quality_metric_tmp
    sim_metric = sim_metric_tmp
    min_jac_score = min_jac_score_tmp
    min_same_parents = min_same_parents_tmp
    min_f1 = min_f1_tmp
    start_time = start_time_tmp
    workers_filter_similar = workers_filter_similar_tmp
    crop_number_in_workers = crop_number_in_workers_tmp

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
global sim_metric
global min_jac_score
global min_f1
global start_time
global workers_filter_similar
global crop_number_in_workers

global y_true_np
global df_dict


def worker_formula_reload(formula_template, expr, summed_expr):
    global min_f1
    best_models = []
    df_columns = df.columns

    for columns in permutations(df_columns, subset_size):
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
            best_models.append(model_info)
        if crop_number_in_workers is not None and len(best_models) >= crop_number_in_workers * 3:
            best_models.sort(key=lambda row: row[quality_metric], reverse=True)
            if workers_filter_similar:
                best_models = similarity_filtering(best_models, sim_metric, min_jac_score, min_same_parents)
            best_models = best_models[:crop_number_in_workers]
            min_f1 = best_models[-1][quality_metric]
    if crop_number_in_workers is not None:
        best_models.sort(key=lambda row: row[quality_metric], reverse=True)
        if workers_filter_similar:
            best_models = similarity_filtering(best_models, sim_metric, min_jac_score, min_same_parents)
        best_models = best_models[:crop_number_in_workers]
    return best_models



def find_best_model_parallel_formula_reload(df, y_true, subset_size, quality_metric, sim_metric, min_jac_score, process_number=2, formula_per_worker=1, \
        file_name='tmp', min_same_parents=1, filter_similar_between_reloads=False, crop_number=None, workers_filter_similar=False, \
        crop_number_in_workers=None, crop_features=None):
    excel_exist = False
    if os.path.exists(f"./Output/BestModels_{file_name}.xlsx"):
        excel_exist = True

    df_columns = df.columns
    columns_number = len(df_columns)
    formulas_number = 2**(2**subset_size) - 2
    models_per_formula = factorial(columns_number) / factorial(columns_number - subset_size)
    total_count = formulas_number * models_per_formula
    print(f'columns number={columns_number}, formulas number={formulas_number}, models per formula={models_per_formula}, total count of models={total_count}')
    formula_batch_size = formula_per_worker * process_number

    if quality_metric == 'f1':
        quality_metric = 'f1_1'
    if quality_metric == 'precision':
        quality_metric = 'precision_1'
    if quality_metric == 'recall':
        quality_metric = 'recall_1'

    # variables (one-charactered) and algebra are needed for simplifying expressions
    variables = list(map(chr, range(122, 122-subset_size,-1)))
    algebra = boolean.BooleanAlgebra()

    start_time = time.time()

    best_models = []
    min_f1 = -1
    finish = False
    formula_batch = []
    formulas_in_batch_count = 0
    overall_formulas_count = 0
    overall_model_count = 0

    for expr in model_string_gen(subset_size):        
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

        simple_tmp = simple_expr.replace('df', 'df_np_cols')
        for i in range(subset_size):
            simple_tmp = simple_tmp.replace(f'columns[{i}]', f'{i}')
        formula_template = eval(f'njit(lambda df_np_cols: {simple_tmp})')

        formula_batch.append((formula_template, simple_expr, summed_expr))
        formulas_in_batch_count += 1
        overall_formulas_count += 1
        if formulas_in_batch_count == formula_batch_size:
            pool = Pool(process_number-1, initializer=worker_init, initargs=(df, y_true, subset_size, quality_metric, sim_metric, min_jac_score, min_same_parents, \
                        min_f1, start_time, workers_filter_similar, crop_number_in_workers))
            print('\nPool started')
            new_models = pool.starmap(worker_formula_reload, formula_batch)
            pool.close()
            pool.join()
            new_models = list(chain.from_iterable(new_models))
            best_models = list(chain.from_iterable([best_models, new_models]))
            if overall_formulas_count == formulas_number:
                finish = True
            if crop_number is not None:
                best_models.sort(key=lambda row: row[quality_metric], reverse=True)
                if filter_similar_between_reloads or finish:
                    best_models = similarity_filtering(best_models, sim_metric, min_jac_score, min_same_parents)
                best_models = best_models[:crop_number+1]
            min_f1 = best_models[-1][quality_metric]
            best_models = add_readable_simple_formulas(best_models, subset_size)

            # Preparing models to be written in excel
            models_to_excel = tupleList_to_df(best_models)
            models_to_excel.drop(['columns_set', 'result'], axis=1, inplace=True)
            beautify_simple(models_to_excel)
            beautify_summed(models_to_excel, subset_size, variables)
            compute_complexity_metrics(models_to_excel)

            models_to_excel = models_to_excel[['tn', 'fp', 'fn', 'tp', 'precision_1', 'recall_1', 'rocauc', 'f1_1', 'accuracy', \
                'elapsed_time', 'columns', 'summed_expr', 'simple_formula', 'number_of_binary_operators', 'max_freq_of_variables']]
            models_to_excel.rename(columns={'precision_1': 'precision', 'recall_1': 'recall', 'f1_1': 'f1'}, inplace=True)

            if excel_exist:
                with pd.ExcelWriter(f"./Output/BestModels_{file_name}.xlsx", mode="a", if_sheet_exists='replace', engine="openpyxl") as writer:
                    models_to_excel.to_excel(writer, sheet_name=f'Size {subset_size}', index=False, freeze_panes=(1,1))
            else:
                models_to_excel.to_excel(f"./Output/BestModels_{file_name}.xlsx", index=False, freeze_panes=(1,1), sheet_name=f'Size {subset_size}')
                excel_exist = True
            overall_model_count += len(formula_batch) * models_per_formula
            formula_batch = []
            formulas_in_batch_count = 0
            print(f'processed {(overall_model_count/total_count) * 100:.1f}% models')

    if not finish:
        pool = Pool(process_number-1, initializer=worker_init, initargs=(df, y_true, subset_size, quality_metric, sim_metric, min_jac_score, min_same_parents, \
                        min_f1, start_time, workers_filter_similar, crop_number_in_workers))
        new_models = pool.starmap(worker_formula_reload, formula_batch)
        pool.close()
        pool.join()
        new_models = list(filter(None, new_models))
        new_models = list(chain.from_iterable(new_models))
        best_models = list(chain.from_iterable([best_models, new_models]))
        best_models.sort(key=lambda row: row[quality_metric], reverse=True)
        best_models = similarity_filtering(best_models, sim_metric, min_jac_score, min_same_parents)
        best_models = add_readable_simple_formulas(best_models, subset_size)

        # Preparing models to be written in excel
        models_to_excel = tupleList_to_df(best_models)
        models_to_excel.drop(['columns_set', 'result'], axis=1, inplace=True)
        beautify_simple(models_to_excel)
        beautify_summed(models_to_excel, subset_size, variables)
        compute_complexity_metrics(models_to_excel)

        models_to_excel = models_to_excel[['tn', 'fp', 'fn', 'tp', 'precision_1', 'recall_1', 'rocauc', 'f1_1', 'accuracy', \
            'elapsed_time', 'columns', 'summed_expr', 'simple_formula', 'number_of_binary_operators', 'max_freq_of_variables']]
        models_to_excel.rename(columns={'precision_1': 'precision', 'recall_1': 'recall', 'f1_1': 'f1'}, inplace=True)

        if excel_exist:
            with pd.ExcelWriter(f"./Output/BestModels_{file_name}.xlsx", mode="a", if_sheet_exists='replace', engine="openpyxl") as writer:
                models_to_excel.to_excel(writer, sheet_name=f'Size {subset_size}', index=False, freeze_panes=(1,1))
        else:
            models_to_excel.to_excel(f"./Output/BestModels_{file_name}.xlsx", index=False, freeze_panes=(1,1), sheet_name=f'Size {subset_size}')
            excel_exist = True
        overall_model_count += len(formula_batch) * models_per_formula
        print(f'processed {(overall_model_count/total_count) * 100:.1f}% models')

    return best_models


######################################################################################################################