import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import os.path
from enum import unique
# import bisect
from itertools import product, combinations, permutations
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


'''General functions'''##############################################################################################

# Generator of boolean formulas in str format using truth tables
def model_string_gen(vars_num):
    inputs = list(product([False, True], repeat=vars_num))
    for output in product([False, True], repeat=len(inputs)):
        terms = []
        for j in range(len(output)):
            if output[j]:
                terms.append(' & '.join(['df[columns[' + str(i) +']]' if input_ else '~df[columns[' + str(i) +']]' for i, input_ in enumerate(inputs[j])]))
        if not terms:
            terms = ['False']
            continue
        expr = ' | '.join(terms)
        yield expr


# Simpify expression with boolean.py library
# In order to simplify and find sums in expression we need to
# replace 'df[columns[{i}]]' with one-character variables
def simplify_expr(expr, subset_size, variables, algebra):
    simple_expr = expr
    for i in range(subset_size):
        simple_expr = simple_expr.replace(f'df[columns[{i}]]', variables[i])
    simple_expr = str(algebra.parse(simple_expr).simplify())
    return simple_expr


# List of tuples (best_models) to dataframe
def tupleList_to_df(best_formulas, in_training=False, parallel=False):
    if in_training:
        models = pd.DataFrame(data=best_formulas, columns=['f1', 'columns', 'expr', 'result'])
    elif parallel:
        models = pd.DataFrame(data=best_formulas, columns=['f1_1', 'f1_0', 'tn', 'fp', 'fn', 'tp', 'precision_1', 'precision_0', 'recall_1', 'recall_0', 'rocauc', 'accuracy', 'elapsed_time', 'summed_expr', 'columns', 'expr', 'simple_formula'])
    else:
        models = pd.DataFrame(data=best_formulas, columns=['precision_0', 'recall_0', 'precision_1', 'recall_1', 'rocauc', 'accuracy', 'f1', 'columns', 'expr', 'result'])
    return models


# Adding to each of tuples of models in best_models human-readable format of expr 
def get_simple_formulas_list(models_list, subset_size):
    new_models_list = []
    # Duplicating expr (last element in tuple) for every model
    for tup in models_list:
        if len(tup) == 17: #hard-coded
            new_models_list.append(tup)
        else:
            new_models_list.append(tup+tuple([tup[-1]]))

    # Replacing 'df[columns[{i}]]' with i-th column name
    for i in range(subset_size):
        for j in range(len(new_models_list)):
            new_models_list[j] = new_models_list[j][:-1] + tuple([new_models_list[j][-1].replace(f'df[columns[{i}]]', new_models_list[j][-3][i])])
    return new_models_list


# Removing duplicates in sorted list, by checking human-readable formula (last element of model tuple)
def removeDuplicateModels(models_list):
    new_models_list = []
    prev = None
    for tup in models_list:
        if prev is None:
            prev = tup[-1]
            continue
        if prev == tup[-1]:
            continue
        new_models_list.append(tup)
        prev = tup[-1]
    return new_models_list


def beautify_simple(df):
    df['simple_formula'] = df.apply(lambda x: x['simple_formula'].replace('~', ' ~'), axis=1)
    df['simple_formula'] = df.apply(lambda x: x['simple_formula'].replace('&', ' & '), axis=1)
    df['simple_formula'] = df.apply(lambda x: x['simple_formula'].replace('|', ' | '), axis=1)


def beautify_summed(df, subset_size, variables):
    df['summed_expr'] = df.apply(lambda x: x['summed_expr'].replace('\'', '') if x['summed_expr'] is not None else None, axis=1)
    df['summed_expr'] = df.apply(lambda x: x['summed_expr'].replace('{', '').replace('}', '') if x['summed_expr'] is not None else None, axis=1)
    for i in range(len(df['columns'][0])):
        df['summed_expr'] = df.apply(lambda x: x['summed_expr'].replace(variables[i], f' {variables[i]} ') if x['summed_expr'] is not None else None, axis=1)
    for i in range(len(df['columns'][0])):
        df['summed_expr'] = df.apply(lambda x: x['summed_expr'].replace(f' {variables[i]} ', x['columns'][i]) if x['summed_expr'] is not None else None, axis=1)


# Compute metrics of formula's complexity (number of binary operations and maximal frequency of variable's occurance)
def compute_complexity_metrics(df):
    def count_operators(s):
        return s.count('|') + s.count('&') + s.count('sum')
    def count_vars(s):
        list_of_vars = re.subn(r'\)>=[0-9]+', '', s)[0].replace('~','').replace('|',',').replace('&',',').replace('sum(','').split(',')
        value_dict = dict((x,list_of_vars.count(x)) for x in set(list_of_vars))
        return max(value_dict.values())
    # df['number_of_binary_operators'] = df['simple_formula'].apply(count_operators)
    df['number_of_binary_operators'] = df.apply(lambda x: count_operators(x['summed_expr']) if x['summed_expr'] is not None else count_operators(x['simple_formula']), axis=1)
    # df['max_freq_of_variables'] = df['simple_formula'].apply(count_vars)
    df['max_freq_of_variables'] = df.apply(lambda x: count_vars(x['summed_expr']) if x['summed_expr'] is not None else count_vars(x['simple_formula']), axis=1)


def sums_generator(subset_size):  
    variables = list(map(chr, range(122, 122-subset_size,-1))) 
    indeces = list(range(0,subset_size,1))  
    variables_with_tilda = variables.copy()  
    sum_dict = {}  
    for i in range(1,subset_size+1):  
        sum_dict[i] = []  
        sum_dict[i].append(list(map(set,list(combinations(variables, i)))))  
        for j in range(1,subset_size+1):  
            indices_comb = list(combinations(indeces, j))  
            for inds in indices_comb:  
                variables_with_tilda = variables.copy()  
                for ind in inds:  
                    variables_with_tilda[ind] = '~'+variables[ind]  
                sum_dict[i].append(list(map(set,list(combinations(variables_with_tilda, i)))))  
          
    return sum_dict  
 
# Take a formula as a string and returns the list of sets in the form of {a,b} if a&b in the formula (DNF-like)   
def formula_partition(s):   
    s = s.replace('(','').replace(')','')  
    list_of_exp = s.split('|')   
    sum_list = []   
    for sub_expr in list_of_exp:   
        if '&' not in sub_expr: 
            sum_list.append(set([sub_expr]))   
        else: 
            sum_list.append(set(sub_expr.split('&')))   
    return sum_list


# Take generated sums and the list generated by formula_partition function (inside) and check if any of generated sums are fully presented in the formula (returns the type of the sum: 1 if sum(a,b,c)>=1, 2 if sum(a,b,c)>=2 and so on)  
def find_one_sum(sum_dict,f):  
      
    sum_list = formula_partition(f)  
    for sum_key in sum_dict.keys():  
        for sub_sum in sum_dict[sum_key]:  
            c = 0  
            len_sum = len(sub_sum)  
            parts_to_replace = []  
            set_of_vars = set()  
            for sub_exp in sub_sum:  
                if sub_exp in sum_list:  
                    c+=1  
                    parts_to_replace.append(sub_exp)  
                    for var in sub_exp:  
                        set_of_vars.add(var)  
            if c==len_sum:  
                if len(parts_to_replace)<len(sum_list):  
                    for part in parts_to_replace:  
                        sum_list.remove(part)  
                    rest_of_formula = str(sum_list)[2:-2].replace('\'','').replace('}, {','|').replace(', ', '&') 
                    sum_formula = 'sum({})>={}'.format(set_of_vars, sum_key) 
                    #formula = 'sum({})>={}|{}'.format(set_of_vars, sum_key, rest_of_formula)  
                else:  
                    #formula = 'sum({})>={}'.format(set_of_vars, sum_key) 
                    rest_of_formula = None 
                    sum_formula = 'sum({})>={}'.format(set_of_vars, sum_key) 
                return sum_formula, rest_of_formula 
             
 
             
def find_sum(sum_dict,f):  
    rest_of_formula = f 
    sums = [] 
    while rest_of_formula!=None and find_one_sum(sum_dict,rest_of_formula)!=None: 
         
        sum_formula, rest_of_formula = find_one_sum(sum_dict,rest_of_formula) 
        sums.append(sum_formula) 
    if len(sums)>0: 
        if rest_of_formula!=None: 
            sums.append(rest_of_formula) 
        return '|'.join(c for c in sums)

######################################################################################################################



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
            tmp_df = tmp_df[~tmp_df.isnull().any(axis=1)]
            y_true_tmp = y_true[~tmp_df.isnull().any(axis=1)]
            try:
                result = model(tmp_df, columns).to_numpy()
            except:
                print(model(tmp_df, columns), columns, expr)

            # Check if model passes theshold
            f1 = f1_score(y_true_tmp, result)
            if f1 > min_f1.value:
                # Calculate metrics
                tn, fp, fn, tp = confusion_matrix(y_true_tmp, result).ravel()
                report_dict = classification_report(y_true_tmp, result, zero_division=0, digits=5, output_dict=True, labels=[0,1])
                rocauc = roc_auc_score(y_true_tmp, result)
                accuracy = accuracy_score(y_true_tmp, result)
                elapsed_time = time.time() - start_time
                model_info = (report_dict['1']['f1-score'], report_dict['0']['f1-score'], tn, fp, fn, tp, report_dict['1']['precision'], report_dict['0']['precision'], report_dict['1']['recall'], report_dict['0']['recall'], rocauc, accuracy, elapsed_time, summed_expr, columns, expr)
                best_models.append(model_info)

        if print_process_logs:
            print(f'Worker {process_name}: Finished processing, resulting in {len(best_models)} models')
        output.put(best_models)
    
    print(f'Worker {process_name}: Stopped working')


# Collector takes best_models (list of tuples) from workers,
# sorts them by f1, removes duplicate models and crop number of models by 1000
def shared_f1_batch_collector(input, file_name, subset_size, min_f1, print_process_logs, batch_size, columns_number):
    process_name = current_process().name
    print(f'Collector {process_name}: Started working')
    # start_time = time.time()
    excel_exist = False
    if os.path.exists(f"./Output/BestModels_{file_name}.xlsx"):
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
        if len(best_models) < 4000:
            best_models.extend(models_table)

        # When number of collected models exceed 4000, then best_models will be sorted
        # and croped by 1000. After that thershold for including in models_table from workers
        # will be updated
        else:
            best_models.extend(models_table)
            best_models = sorted(best_models, key=lambda tup: tup[0], reverse=True)
            best_models = get_simple_formulas_list(best_models, subset_size)
            best_models = removeDuplicateModels(best_models)
            best_models = best_models[:1000]
            min_f1.value = best_models[-1][0]

            # Preparing models to be written in excel
            models_to_excel = tupleList_to_df(best_models, parallel=True)
            beautify_simple(models_to_excel)
            beautify_summed(models_to_excel, subset_size, variables)
            compute_complexity_metrics(models_to_excel)

            if excel_exist:
                with pd.ExcelWriter(f"./Output/BestModels_{file_name}.xlsx", mode="a", if_sheet_exists='replace', engine="openpyxl") as writer:
                    models_to_excel.to_excel(writer, sheet_name=f'Size {subset_size}', index=False, freeze_panes=(1,1))
            else:
                models_to_excel.to_excel(f"./Output/BestModels_{file_name}.xlsx", index=False, freeze_panes=(1,1), sheet_name=f'Size {subset_size}')
                excel_exist = True

            if print_process_logs:
                print(f'Collector {process_name}: Sorted best models')

    # Case when got 'STOP' before sorting last batches of models
    best_models = sorted(best_models, key=lambda tup: tup[0], reverse=True)
    best_models = get_simple_formulas_list(best_models, subset_size)
    best_models = removeDuplicateModels(best_models)
    best_models = best_models[:1000]

    models_to_excel = tupleList_to_df(best_models, parallel=True)
    beautify_simple(models_to_excel)
    beautify_summed(models_to_excel, subset_size, variables)
    compute_complexity_metrics(models_to_excel)

    if excel_exist:
        with pd.ExcelWriter(f"./Output/BestModels_{file_name}.xlsx", mode="a", if_sheet_exists='replace', engine="openpyxl") as writer:
            models_to_excel.to_excel(writer, sheet_name=f'Size {subset_size}', index=False, freeze_panes=(1,1))
    else:
        models_to_excel.to_excel(f"./Output/BestModels_{file_name}.xlsx", index=False, freeze_panes=(1,1), sheet_name=f'Size {subset_size}')
    print(f'Collector {process_name}: Stopped working')


def find_best_model_shared_f1_batch_parallel(df, y_true, subset_size, process_number=False, print_process_logs=False, batch_size=False, file_name='tmp'):
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
    collector = Process(target=shared_f1_batch_collector, args=(done_queue, file_name, subset_size, min_f1, print_process_logs, batch_size, len(df_columns)))
    collector.start()

    # Creating batches (expr, columns_batch), where expr is formula and
    # columns_batch is list of subsets of columns, i.e. batch consists of several models with
    # same formula, but different subsets
    for expr in tqdm(model_string_gen(subset_size), total=formulas_number-1):
        columns_batch = []
        count = 0
        
        simple_expr = simplify_expr(expr, subset_size, variables, algebra)
        # for i in range(subset_size):
        #     simple_expr = simple_expr.replace(f'df[columns[{i}]]', variables[i])
        # simple_expr = str(algebra.parse(simple_expr).simplify())

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