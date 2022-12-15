import os.path
from itertools import combinations, chain, islice
import time
from math import factorial
from copy import deepcopy

import pandas as pd
import numpy as np
from numba import njit
import boolean

from multiprocessing import Queue, Pool, cpu_count, Process, current_process
#from multiprocess import Queue, Pool, cpu_count, Process, current_process
from multiprocessing.sharedctypes import Value
#from multiprocess.sharedctypes import Value

from utils import model_string_gen, simplify_expr, add_readable_simple_formulas, find_sum, sums_generator
from utils import list_to_df, beautify_simple, beautify_summed, similarity_filtering, post_simplify
from metrics_utils import count_confusion_matrix, get_parent_set, count_operators, count_vars, count_actual_subset_size
from metrics_utils import calculate_metrics_for, negate_model


def worker_init(df_dict_tmp, y_true_tmp, all_formulas_tmp, columns_ordered_tmp, subset_size_tmp, quality_metric_tmp, sim_metric_tmp, \
                crop_number_in_workers_tmp, excessive_models_num_coef_tmp):
    global df_dict
    global y_true
    global all_formulas
    global columns_ordered
    global subset_size
    global quality_metric
    global sim_metric
    global start_time
    global crop_number_in_workers
    global excessive_models_num_coef

    df_dict = df_dict_tmp
    y_true = y_true_tmp
    all_formulas = all_formulas_tmp
    columns_ordered = columns_ordered_tmp
    subset_size = subset_size_tmp
    quality_metric = quality_metric_tmp
    sim_metric = sim_metric_tmp
    start_time = time.time()
    crop_number_in_workers = crop_number_in_workers_tmp
    excessive_models_num_coef = excessive_models_num_coef_tmp


global df_dict
global y_true
global all_formulas
global columns_ordered
global subset_size
global quality_metric
global sim_metric
global start_time
global crop_number_in_workers
global excessive_models_num_coef


# @log_sparse
@njit
def make_nan_mask(nan_mask, df_nan_cols, subset_size):
    for i in range(subset_size):
        nan_mask &= df_nan_cols[i]
    return nan_mask


# @log_sparse
@njit
def apply_nan_mask_list(pred, nan_mask):
    for i in range(len(pred)):
        if nan_mask[i]:
            pred[i] = np.nan
    return pred


def worker(start_idx, end_idx, min_quality):
    best_models = []
    for columns in islice(combinations(columns_ordered, subset_size), start_idx, end_idx):
        columns = list(columns)
        nan_mask = np.full_like(y_true, True, dtype=bool)
        df_np_cols = []
        df_nan_cols = []
        for col in columns:
            df_np_cols.append(df_dict[col]['data'])
            df_nan_cols.append(df_dict[col]['nan_mask'])
        df_np_cols = np.array(df_np_cols)
        df_nan_cols = np.array(df_nan_cols)

        for formula_dict in all_formulas:
            formula_template = formula_dict['formula_template']
            expr = formula_dict['expr']
            summed_expr = formula_dict['summed_expr']
            result = formula_template(df_np_cols)
            nan_mask = make_nan_mask(nan_mask, df_nan_cols, subset_size)
            nan_count = np.count_nonzero(nan_mask)
            result = apply_nan_mask_list(result, nan_mask)

            tp, fp, fn, tn = count_confusion_matrix(y_true, result)
            precision, recall, f1, rocauc, accuracy = calculate_metrics_for(tp, fp, fn, tn)


            # precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
            # recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
            # f1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
            # fpr = 0 if (fp + tn) == 0 else fp / (fp + tn)
            # rocauc = (1 + recall - fpr) / 2
            # accuracy = 0 if (tp + fp + fn + tn) == 0 else (tp + tn) / (tp + fp + fn + tn)
            if quality_metric == 'f1_1':
                quality = f1
            elif quality_metric == 'precision_1':
                quality = precision
            elif quality_metric == 'recall_1':
                quality = recall
            elif quality_metric == 'rocauc':
                quality = rocauc
            elif quality_metric == 'accuracy':
                quality = accuracy

            if quality > min_quality:
                precision_0 = 0 if (tn + fn) == 0 else tn / (tn + fn)
                recall_0 = 0 if (tn + fp) == 0 else tn / (tn + fp)
                f1_0 = 0 if (precision_0 + recall_0) == 0 else 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
                columns_set = get_parent_set(columns)
                elapsed_time = time.time() - start_time
                simple_formula = expr
                for i in range(subset_size):
                    simple_formula = simple_formula.replace(f'df_np_cols[{i}]', columns[i])
                model_info = {'f1_1': f1, 'f1_0': f1_0, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'precision_1': precision, 'precision_0': precision_0,\
                    'recall_1': recall, 'recall_0': recall_0, 'rocauc': rocauc, 'accuracy': accuracy, 'elapsed_time': elapsed_time, 'nan_ratio': nan_count/y_true.shape[0], \
                    'summed_expr': summed_expr, 'columns': columns, 'expr': expr, 'expr_len': len(expr), 'is_negated': 'False',  'simple_formula': simple_formula, 'columns_set': columns_set, \
                    'number_of_binary_operators': formula_dict['number_of_binary_operators'], 'max_freq_of_variables': formula_dict['max_freq_of_variables']}
                if sim_metric == 'JAC_SCORE':
                    model_info['result'] = result
                else:
                    model_info['result'] = None
                best_models.append(model_info)
            result_neg = negate_model(result)
            tp, fp, fn, tn = count_confusion_matrix(y_true, result_neg)
            precision, recall, f1, rocauc, accuracy = calculate_metrics_for(tp, fp, fn, tn)
            if quality_metric == 'f1_1':
                quality = f1
            elif quality_metric == 'precision_1':
                quality = precision
            elif quality_metric == 'recall_1':
                quality = recall
            elif quality_metric == 'rocauc':
                quality = rocauc
            elif quality_metric == 'accuracy':
                quality = accuracy
            if quality > min_quality:
                precision_0 = 0 if (tn + fn) == 0 else tn / (tn + fn)
                recall_0 = 0 if (tn + fp) == 0 else tn / (tn + fp)
                f1_0 = 0 if (precision_0 + recall_0) == 0 else 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
                columns_set = get_parent_set(columns)
                elapsed_time = time.time() - start_time
                simple_formula = expr
                for i in range(subset_size):
                    simple_formula = simple_formula.replace(f'df_np_cols[{i}]', columns[i])
                expr = '~(' + expr + ')'
                simple_formula = '~(' + simple_formula + ')'
                model_info = {'f1_1': f1, 'f1_0': f1_0, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'precision_1': precision, 'precision_0': precision_0,\
                    'recall_1': recall, 'recall_0': recall_0, 'rocauc': rocauc, 'accuracy': accuracy, 'elapsed_time': elapsed_time, 'nan_ratio': nan_count/y_true.shape[0], \
                    'summed_expr': summed_expr, 'columns': columns, 'expr': expr, 'expr_len': len(expr), 'is_negated': 'True', 'simple_formula': simple_formula, 'columns_set': columns_set, \
                    'number_of_binary_operators': formula_dict['number_of_binary_operators'], 'max_freq_of_variables': formula_dict['max_freq_of_variables']}
                if sim_metric == 'JAC_SCORE':
                    model_info['result'] = result
                else:
                    model_info['result'] = None
                best_models.append(model_info)


            if crop_number_in_workers is not None and len(best_models) >= crop_number_in_workers * excessive_models_num_coef:
                best_models.sort(key=lambda row: (row[quality_metric], -row['number_of_binary_operators']), reverse=True)
                best_models = best_models[:crop_number_in_workers]
                min_quality = best_models[-1][quality_metric]

    return best_models


def find_best_models(df, y_true, columns_ordered, subset_size, quality_metric, sim_metric, min_jac_score, min_same_parents=1, process_number=10,\
                        crop_number=None, crop_number_in_workers=1000, excessive_models_num_coef=3, batch_size=10000, \
                        filter_similar_between_reloads=True, file_name='tmp', crop_features=None, dataset_frac=None, incremental_run=False, \
                        crop_features_after_size=1):
    excel_exist = False
    if os.path.exists(f"./Output/BestModels_{file_name}.xlsx"):
        excel_exist = True

    columns_number = len(columns_ordered)
    subset_number = int(factorial(columns_number) / factorial(columns_number - subset_size))

    if batch_size > subset_number:
        batch_size = subset_number

    if quality_metric == 'f1':
        quality_metric = 'f1_1'
    if quality_metric == 'precision':
        quality_metric = 'precision_1'
    if quality_metric == 'recall':
        quality_metric = 'recall_1'

    # variables (one-charactered) and algebra are needed for simplifying expressions
    variables = list(map(chr, range(122, 122-subset_size,-1)))
    algebra = boolean.BooleanAlgebra()

    df_dict = {}
    for col in df.columns:
        df_dict[col] = {'data': df[col].values.astype(bool), 'nan_mask': pd.isna(df[col]).values}
    y_true = y_true.values.astype(float)

    all_formulas = []
    for expr in model_string_gen(subset_size):        
        simple_expr = simplify_expr(expr, subset_size, variables, algebra)
        # If formula is a tautology
        if simple_expr == '1':
            continue
        actual_subset_size = count_actual_subset_size(simple_expr)
        if actual_subset_size < subset_size:
            continue
        number_of_binary_operators = count_operators(simple_expr)
        max_freq_of_variables = count_vars(simple_expr)

        summed_expr = find_sum(sums_generator(subset_size), simple_expr)

        # Replace one-charachter variables in simplified expr with 'df[columns[{i}]]'
        # for easy execution
        for i in range(subset_size):
            simple_expr = simple_expr.replace(variables[i], f'df_np_cols[{i}]')

        all_formulas.append({
            'formula_template': eval(f'njit(lambda df_np_cols: {simple_expr})'),
            'expr': simple_expr,
            'summed_expr': summed_expr,
            'number_of_binary_operators': number_of_binary_operators,
            'max_freq_of_variables': max_freq_of_variables
        })
    formulas_number = len(all_formulas)
    total_model_count = formulas_number * subset_number
    print(f'formulas_number: {formulas_number}  columns_subset_number(models_per_formula): {subset_number}  total_count_of_models: {total_model_count}')

    worker_timer = time.time()
    pool = Pool(process_number, initializer=worker_init, initargs=(df_dict, y_true, all_formulas, columns_ordered, subset_size, quality_metric, \
                sim_metric, crop_number_in_workers, excessive_models_num_coef))
    print(f'number of workers launched: {process_number}  time to launch: {time.time() - worker_timer}')

    start_time = time.time()
    best_models = []
    best_models_not_filtered = []
    min_quality = -1
    current_model_count = 0
    batch = []
    last_reload = False
    for subset_index in range(0, subset_number, batch_size):
        if subset_number - subset_index <= batch_size:
            batch.append((subset_index, subset_number, min_quality))
            current_model_count += (subset_number - subset_index) * formulas_number
            last_reload = True
        else:
            batch.append((subset_index, subset_index+batch_size, min_quality))
            current_model_count += batch_size * formulas_number
        
        if len(batch) >= process_number or last_reload:
            # print(batch)
            # print('Workers time')
            new_models = pool.starmap(worker, batch)
            new_models = list(chain.from_iterable(new_models))
            best_models = list(chain.from_iterable([best_models, new_models]))
            # print('got models')
            if crop_number is not None or last_reload:
                best_models.sort(key=lambda row: (row[quality_metric], -row['number_of_binary_operators']), reverse=True)
            if filter_similar_between_reloads or last_reload:
                best_models_not_filtered = deepcopy(best_models)
                best_models = similarity_filtering(best_models, sim_metric, min_jac_score, min_same_parents)
            if crop_number is not None and not last_reload:
                best_models = best_models[:crop_number+1]
            min_quality = best_models[-1][quality_metric]
            # print('sorted & filtered models')

            models_to_excel = list_to_df(best_models)
            models_to_excel.drop(['columns_set', 'result'], axis=1, inplace=True)
            # models_to_excel = post_simplify(models_to_excel, subset_size, variables, algebra)
            models_to_excel['simple_formula'] = models_to_excel.apply(lambda x: post_simplify(x, subset_size, variables, algebra), axis=1)
            beautify_simple(models_to_excel)
            beautify_summed(models_to_excel, subset_size, variables)

            models_to_excel = models_to_excel[['tn', 'fp', 'fn', 'tp', 'precision_1', 'recall_1', 'rocauc', 'f1_1', 'nan_ratio', 'accuracy', \
                'elapsed_time', 'columns', 'summed_expr', 'is_negated', 'simple_formula', 'number_of_binary_operators', 'max_freq_of_variables']]
            models_to_excel.rename(columns={'precision_1': 'precision', 'recall_1': 'recall', 'f1_1': 'f1'}, inplace=True)

            if excel_exist:
                with pd.ExcelWriter(f"./Output/BestModels_{file_name}.xlsx", mode="a", if_sheet_exists='replace', engine="openpyxl") as writer:
                    models_to_excel.to_excel(writer, sheet_name=f'Size {subset_size}', index=False, freeze_panes=(1,1))
            else:
                models_to_excel.to_excel(f"./Output/BestModels_{file_name}.xlsx", index=False, freeze_panes=(1,1), sheet_name=f'Size {subset_size}')
                excel_exist = True

            # print('wrote models to excel')
            
            batch = []
            elapsed_time = time.time() - start_time
            elapsed_time_per_model = elapsed_time/current_model_count
            print(f'processed_models: {current_model_count/total_model_count * 100:.1f}%  elapsed_seconds: {elapsed_time:.2f}  current_quality_threshold: {min_quality:.2f}  estimated_seconds_remaining: {(total_model_count - current_model_count) * elapsed_time_per_model:.2f}')
    worker_timer = time.time()
    pool.close()
    pool.join()
    print(f'number of workers terminated: {process_number}  time to terminate: {time.time() - worker_timer}')
    return best_models_not_filtered