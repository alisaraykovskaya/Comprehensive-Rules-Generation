import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from numba import vectorize

from itertools import product, combinations, permutations
import os.path
import re

import fastmetrics
import boolean


def similarity_filtering(best_models, metric, min_jac_score, min_same_parents):
    i = 0

    while i < len(best_models):
        for j in range(len(best_models)-1, i, -1):
            if compare_model_similarity(best_models[i]['result'], best_models[j]['result'], best_models[i]['columns_set'], \
                                        best_models[j]['columns_set'], metric, min_jac_score, min_same_parents):
                del best_models[j]
        i += 1
    return best_models


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


# Adding to each of tuples of models in best_models human-readable format of expr 
def get_simple_formulas_list(models_list, subset_size):
    new_models_list = []
    # Duplicating expr (last element in tuple) for every model
    for tup in models_list:
        if len(tup) == 19: #hard-coded
            new_models_list.append(tup)
        else:
            # print(len(tup))
            new_models_list.append(tup+tuple([tup[-3]]))

    # Replacing 'df[columns[{i}]]' with i-th column name
    for i in range(subset_size):
        for j in range(len(new_models_list)):
            new_models_list[j] = new_models_list[j][:-1] + tuple([new_models_list[j][-1].replace(f'df[columns[{i}]]', new_models_list[j][-5][i])])
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


# List of tuples (best_models) to dataframe
def tupleList_to_df(best_formulas, all_in1=False, parallel_continuous=False, reload=False):
    if all_in1:
        models = pd.DataFrame(data=best_formulas, columns=['precision_0', 'recall_0', 'precision_1', 'recall_1', 'rocauc', 'accuracy', 'f1', 'columns', 'expr', 'result'])
    elif parallel_continuous:
        models = pd.DataFrame(data=best_formulas, columns=['f1_1', 'f1_0', 'tn', 'fp', 'fn', 'tp', 'precision_1', 'precision_0', 'recall_1', 'recall_0', 'rocauc', 'accuracy', 'elapsed_time', 'summed_expr', 'columns', 'expr', 'columns_set', 'result', 'simple_formula'])
    elif reload:
        models = pd.DataFrame.from_records(best_formulas)
    return models


def add_readable_simple_formulas(best_models, subset_size):
    for j in range(len(best_models)):
        best_models[j]['simple_formula'] = best_models[j]['expr']
        for i in range(subset_size):
            best_models[j]['simple_formula'] = best_models[j]['simple_formula'].replace(f'df[columns[{i}]]', best_models[j]['columns'][i])
    return best_models


@vectorize
def generate_confusion_matrix(x, y):
    """
    NumPy ufunc implemented with Numba that generates a confusion matrix as follows:
    1 = True Positive, 2 = False Positive, 3 = False Negative, 4 = True Negative.
    """
    if x and y:
        return 1

    elif not x and y:
        return 2

    elif x and not y:
        return 3

    else:
        return 4


def count_confusion_matrix(y_true, y_pred):
    matrix = generate_confusion_matrix(y_true, y_pred)
    tp = np.count_nonzero(matrix == 1)  # True Positive
    fp = np.count_nonzero(matrix == 2)  # False Positive
    fn = np.count_nonzero(matrix == 3)  # False Negative
    tn = np.count_nonzero(matrix == 4)  # True Negative
    return tp, fp, fn, tn



# Function to obtain a set of parent features of a model --- add to 'model info'?
def get_parent_set(list_of_columns):
    return set(list(map(get_parent_feature, list_of_columns)))

# Function to get parent feature's name from its binary name
def get_parent_feature(column_name):
     if '∈' in column_name:
         return column_name[:column_name.index('∈')]
     elif '<=' in column_name:
         return column_name[:column_name.index('<=')]
     elif '=' in column_name:
         return column_name[:column_name.index('=')]
     else:
         return column_name

def parent_features_similarity(parent_set1, parent_set2, threshold = 1):
    if len(parent_set1.intersection(parent_set2))>=threshold:
        return True
    else:
        return False

# Returns True if two models are so similar one should be filtered, False otherwise
# Parameter metric is a string describing the similarity metric to be used
def compare_model_similarity(model1, model2, columns1, columns2, metric, min_jac_score=0.9, min_same_parents=1):
    if metric == "JAC_SCORE":
        return fastmetrics.fast_jaccard_score(model1, model2) >= min_jac_score
    elif metric == 'PARENT':
        # print(columns1, columns2)
        return parent_features_similarity(columns1, columns2, min_same_parents)


def log_exec(file_name, execution_type, sim_metric, subset_size, rows_num, cols_num, elapsed_time, process_number, batch_size):
    if os.path.exists("./Output/log.xlsx"):
        log = pd.read_excel('./Output/log.xlsx')
        search_idx = log.loc[(log['dataset'] == file_name) & (log['execution_type'] == execution_type) & (log['sim_metric'] == sim_metric) & \
            (log['subset_size'] == subset_size) & (log['rows_num'] == rows_num) & (log['cols_num'] == cols_num) & \
            (log['process_number'] == process_number) & (log['batch_size'] == batch_size)].index.tolist()
        if len(search_idx) == 1:
            log.loc[search_idx, ['elapsed_time']] = elapsed_time
            with pd.ExcelWriter('./Output/log.xlsx', mode="w", engine="openpyxl") as writer:
                log.to_excel(writer, sheet_name='Logs', index=False, freeze_panes=(1,1))
        else:
            new_row = pd.DataFrame(data={'dataset': [file_name], 'execution_type': [execution_type], 'sim_metric': [sim_metric], \
                'subset_size': [subset_size], 'rows_num': [rows_num], 'cols_num': [cols_num], 'elapsed_time': [elapsed_time], \
                'process_number': [process_number], 'batch_size': [batch_size]})
            log = pd.concat([log, new_row], ignore_index=True)
            with pd.ExcelWriter('./Output/log.xlsx', mode="w", engine="openpyxl") as writer:
                log.to_excel(writer, sheet_name='Logs', index=False, freeze_panes=(1,1))
    else:
        log = pd.DataFrame(data={'dataset': [file_name], 'execution_type': [execution_type], 'sim_metric': [sim_metric], \
            'subset_size': [subset_size], 'rows_num': [rows_num], 'cols_num': [cols_num], 'elapsed_time': [elapsed_time], 'process_number': [process_number], 'batch_size': [batch_size]})
        with pd.ExcelWriter('./Output/log.xlsx', mode="w", engine="openpyxl") as writer:
                log.to_excel(writer, sheet_name='Logs', index=False, freeze_panes=(1,1))


def compute_metrics(y_true, result, compute_f1=False):
    precision_0 = precision_score(y_true, result, pos_label=0, zero_division=0)
    recall_0 = recall_score(y_true, result, pos_label=0)
    precision_1 = precision_score(y_true, result, zero_division=0)
    recall_1 = recall_score(y_true, result)
    rocauc = roc_auc_score(y_true, result)
    accuracy = accuracy_score(y_true, result)
    if compute_f1:
        f1 = f1_score(y_true, result)
        return precision_0, recall_0, precision_1, recall_1, rocauc, accuracy, f1
    return precision_0, recall_0, precision_1, recall_1, rocauc, accuracy


def validate_model(columns, expr, X_test, y_test):
    model = eval('lambda df, columns: ' + expr)
    result = model(X_test, columns)
    metrics = [compute_metrics(y_test, result, compute_f1=True)]
    metrics = pd.DataFrame(data=metrics, columns=['precision_0', 'recall_0', 'precision_1', 'recall_1', 'rocauc', 'accuracy', 'f1'])
    return metrics