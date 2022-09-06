import pandas as pd
import numpy as np
from tqdm import tqdm

import os.path
from enum import unique
from itertools import permutations
import time
from math import factorial

from multiprocessing import Queue, Pool, cpu_count, Process, current_process
#from multiprocess import Queue, Pool, cpu_count, Process, current_process
from multiprocessing.sharedctypes import Value
#from multiprocess.sharedctypes import Value
import openpyxl as pxl
import boolean

from utils import get_parent_set, compare_model_similarity, count_confusion_matrix, get_simple_formulas_list, removeDuplicateModels
from utils import tupleList_to_df, beautify_simple, beautify_summed, compute_complexity_metrics, model_string_gen, simplify_expr
from utils import find_sum, sums_generator

'''Shared f1 batch parallel execution'''#############################################################################

# Worker takes models (expression, expression with sums, columns batch) from queue, then evaluates all 
# models (expression + one columns subset in batch). If model f1_score is better that threshold (min_f1)
# then model is appended in best_models, which will be put in queue to collector
# Here min_f1 is a shared variable which will be updated by collector
def shared_f1_batch_worker(input, output, df, y_true, min_f1, start_time, print_process_logs):
    process_name = current_process().name
    print(f'Worker {process_name}: Started working')

    # Read expr (formula), columns_batch from input (task_queue), until get 'STOP'
    for expr, summed_expr, columns_batch in iter(input.get, 'STOP'):

        # best_models - list of models, which passed thershold min_f1
        # Each of the models consists of metrics, elapsed time, columns subset and formula
        best_models = []

        # model is a lamda function, which generated using eval() and expr
        model = eval('lambda df, columns: ' + expr)

        if print_process_logs:
            print(f'Worker {process_name}: Started processing {len(columns_batch)} models, with min_f1={min_f1.value}')

        # Evaluate models from all subsets of columns in batch
        for i in range(len(columns_batch)):
            # Creating tmp_df with needed columns only
            # and removing all Nans from tmp_df, by this we ensure that
            # we didn't remove too much rows
            columns = list(columns_batch[i])
            tmp_df = df[columns]
            tmp_idx = ~tmp_df.isnull().any(axis=1)
            tmp_df = tmp_df[tmp_idx]
            y_true_tmp = y_true[tmp_idx].values
            try:
                result = model(tmp_df, columns).to_numpy()
            except:
                print(model(tmp_df, columns), columns, expr)

            tp, fp, fn, tn = count_confusion_matrix(y_true_tmp, result)
            precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
            recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
            # Check if model passes theshold
            f1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
            # f1 = f1_score(y_true_tmp, result)
            if f1 > min_f1.value:
                # Calculate metrics
                # tn, fp, fn, tp = confusion_matrix(y_true_tmp, result).ravel()
                # report_dict = classification_report(y_true_tmp, result, zero_division=0, digits=5, output_dict=True, labels=[0,1])
                # rocauc = roc_auc_score(y_true_tmp, result)
                # accuracy = accuracy_score(y_true_tmp, result)

                fpr = 0 if (fp + tn) == 0 else fp / (fp + tn)
                rocauc = (1 + recall - fpr) / 2
                accuracy = 0 if (tp + fp + fn + tn) == 0 else (tp + tn) / (tp + fp + fn + tn)
                precision_0 = 0 if (tn + fn) == 0 else tn / (tn + fn)
                recall_0 = 0 if (tn + fp) == 0 else tn / (tn + fp)
                f1_0 = 0 if (precision_0 + recall_0) == 0 else 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
                columns_set = get_parent_set(columns)
                elapsed_time = time.time() - start_time
                model_info = (f1, f1_0, tn, fp, fn, tp, precision, precision_0, recall, recall_0, rocauc, accuracy, elapsed_time, \
                    summed_expr, columns, expr, columns_set, result)
                # model_info = (report_dict['1']['f1-score'], report_dict['0']['f1-score'], tn, fp, fn, tp, report_dict['1']['precision'], \
                #     report_dict['0']['precision'], report_dict['1']['recall'], report_dict['0']['recall'], rocauc, accuracy, elapsed_time, summed_expr, columns, expr)
                best_models.append(model_info)

        if print_process_logs:
            print(f'Worker {process_name}: Finished processing, resulting in {len(best_models)} models')
        output.put(best_models)
    
    print(f'Worker {process_name}: Stopped working')


# Collector takes best_models (list of tuples) from workers,
# sorts them by f1, removes duplicate models and crop number of models by 1000
def shared_f1_batch_collector(input, file_name, subset_size, metric, min_jac_score, min_f1, print_process_logs, batch_size, columns_number):
    process_name = current_process().name
    print(f'Collector {process_name}: Started working')
    # start_time = time.time()
    excel_exist = False
    if os.path.exists(f"./Output/BestModels_{file_name}_pbc.xlsx"):
        excel_exist = True
    variables = list(map(chr, range(122, 122-subset_size,-1)))

    # Parameters for printing progress
    formulas_number = 2**(2**subset_size)
    # subset_number = comb(columns_number, subset_size)
    subset_number = factorial(columns_number) / factorial(columns_number - subset_size)
    total_count = formulas_number * subset_number
    percent_thr = 0.1
    processed_count = 0

    # best_models - (sorted) list of best models
    # Each of the models consists of metrics, elapsed time, columns subset and formula
    best_models = []

    start_time = time.time()

    # Read models_table from input (done_queue), until get 'STOP'
    # models_table - list of models from workers in the same format as best_models
    for models_table in iter(input.get, 'STOP'):

        # Printing progress every 10 %
        processed_count += batch_size
        if processed_count >= total_count * percent_thr:
            # print(f'Collected {percent_thr*100}% of models, elapsed_time={time.time() - start_time}')
            # percent_thr += 0.1
            print(f'Collected {percent_thr*100}% of models')
            percent_thr += 0.1
            start_time = time.time()

        current_time = time.time()
        if current_time - start_time > 3*60:
            current_percent = processed_count / total_count
            print(f'Collected {current_percent*100}% of models')
            start_time = time.time()

        # Collector will be adding models from workers to best_models, until number of models exceed 4000
        if len(best_models) < 20000:
            best_models.extend(models_table)

        # When number of collected models exceed 4000, then best_models will be sorted
        # and croped by 1000. After that thershold for including in models_table from workers
        # will be updated
        else:
            best_models.extend(models_table)
            best_models = sorted(best_models, key=lambda tup: tup[0], reverse=True)
            best_models = get_simple_formulas_list(best_models, subset_size)
            best_models = removeDuplicateModels(best_models)
            best_models = best_models[:2000]
            min_f1.value = best_models[-1][0]

            # Preparing models to be written in excel
            models_to_excel = tupleList_to_df(best_models, parallel_continuous=True)
            models_to_excel.drop(['columns_set', 'result'], axis=1, inplace=True)
            # print(models_to_excel)
            beautify_simple(models_to_excel)
            beautify_summed(models_to_excel, subset_size, variables)
            compute_complexity_metrics(models_to_excel)

            if excel_exist:
                with pd.ExcelWriter(f"./Output/BestModels_{file_name}_pbc.xlsx", mode="a", if_sheet_exists='replace', engine="openpyxl") as writer:
                    models_to_excel.to_excel(writer, sheet_name=f'Size {subset_size}', index=False, freeze_panes=(1,1))
            else:
                models_to_excel.to_excel(f"./Output/BestModels_{file_name}_pbc.xlsx", index=False, freeze_panes=(1,1), sheet_name=f'Size {subset_size}')
                excel_exist = True

            if print_process_logs:
                print(f'Collector {process_name}: Sorted best models')

    # Case when got 'STOP' before sorting last batches of models
    best_models = sorted(best_models, key=lambda tup: tup[0], reverse=True)
    # i = 0
    # while i < len(best_models):
    #     j = i+1

    #     while j < len(best_models):
    #         if len(best_models[i]) == 18:
    #             res1 = best_models[i][-1]
    #             columns1 = best_models[i][-2]
    #         else:
    #             res1 = best_models[i][-2]
    #             columns1 = best_models[i][-3]
    #         if len(best_models[j]) == 18:
    #             res2 = best_models[i][-1]
    #             columns2 = best_models[i][-2]
    #         else:
    #             res2 = best_models[i][-2]
    #             columns2 = best_models[i][-3]
    #         print(type(res1), type(res2))
    #         if compare_model_similarity(res1, res2, columns1, columns2, metric, min_jac_score):
    #             del best_models[j]
    #             j -= 1

    #         j += 1

    #     i += 1
    best_models = get_simple_formulas_list(best_models, subset_size)
    best_models = removeDuplicateModels(best_models)
    # best_models = best_models[:1000]

    models_to_excel = tupleList_to_df(best_models, parallel_continuous=True)
    models_to_excel.drop(['columns_set', 'result'], axis=1, inplace=True)
    # print(models_to_excel)
    beautify_simple(models_to_excel)
    beautify_summed(models_to_excel, subset_size, variables)
    compute_complexity_metrics(models_to_excel)

    if excel_exist:
        with pd.ExcelWriter(f"./Output/BestModels_{file_name}_pbc.xlsx", mode="a", if_sheet_exists='replace', engine="openpyxl") as writer:
            models_to_excel.to_excel(writer, sheet_name=f'Size {subset_size}', index=False, freeze_panes=(1,1))
    else:
        models_to_excel.to_excel(f"./Output/BestModels_{file_name}_pbc.xlsx", index=False, freeze_panes=(1,1), sheet_name=f'Size {subset_size}')
    print(f'Collector {process_name}: Stopped working')


def find_best_model_shared_f1_batch_parallel(df, y_true, subset_size, metric, min_jac_score, process_number=False, print_process_logs=False, batch_size=False, file_name='tmp'):
    formulas_number = 2**(2**subset_size)
    df_columns = df.columns

    # variables (one-charactered) and algebra are needed for simplifying expressions
    variables = list(map(chr, range(122, 122-subset_size,-1)))
    algebra = boolean.BooleanAlgebra()

    # Workers will be taking batch of models to evaluate from task_queue
    task_queue = Queue(maxsize=5000)
    # Result of workers' evaluating will be given to collector through done_queue
    done_queue = Queue(maxsize=5000)
    # Parallel-safe threshold for model to be in top best models
    min_f1 = Value('d', -1.0)
    # set of simplified expressions, because no need of processing same expressions
    expr_set = set()

    if not process_number:
        process_number = cpu_count()-5
    if not batch_size:
        batch_size = 1000
    start_time = time.time()

    # Workers and collector start
    pool = Pool(process_number-1, initializer=shared_f1_batch_worker, initargs=(task_queue, done_queue, df, y_true, min_f1, start_time, print_process_logs))
    collector = Process(target=shared_f1_batch_collector, args=(done_queue, file_name, subset_size, metric, min_jac_score, min_f1, print_process_logs, batch_size, len(df_columns)))
    collector.start()

    # Creating batches (expr, columns_batch), where expr is formula and
    # columns_batch is list of subsets of columns, i.e. batch consists of several models with
    # same formula, but different subsets
    for expr in tqdm(model_string_gen(subset_size), total=formulas_number-1):
        columns_batch = []
        count = 0
        
        simple_expr = simplify_expr(expr, subset_size, variables, algebra)
        summed_expr = find_sum(sums_generator(subset_size), simple_expr)

        # Replace one-charachter variables in simplified expr with 'df[columns[{i}]]'
        # for easy execution
        for i in range(subset_size):
            simple_expr = simple_expr.replace(variables[i], f'df[columns[{i}]]')

        # If formula is a tautology
        if simple_expr == '1':
            continue

        # Check if expr already processed
        if simple_expr not in expr_set:
            expr_set.add(simple_expr)
        else:
            continue

        # Columns batch creation and giving to workers
        for columns in permutations(df_columns, subset_size):
            if count > batch_size - 1:
                task_queue.put((simple_expr, summed_expr, columns_batch))
                columns_batch = []
                count = 0
            columns_batch.append(columns)
            count += 1
        if count <= batch_size - 1:
            task_queue.put((simple_expr, summed_expr, columns_batch))

    # Stop workers
    for _ in range(process_number):
        task_queue.put('STOP')
    pool.close()
    pool.join()
    
    # Stop collector
    done_queue.put('STOP')
    collector.join()

    elapsed_time = time.time() - start_time
    print(f'Elapsed time {elapsed_time} seconds')
    return


######################################################################################################################