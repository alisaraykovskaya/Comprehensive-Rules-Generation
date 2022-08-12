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
from utils import validate_model, compare_model_similarity


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

             
def find_sum(sum_dict, f):
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


def bisect_left(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x) will
    insert just before the leftmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if key(a[mid]) < x:
                hi = mid
            else:
                lo = mid + 1
    return lo


# @profile
def check_model_perfomance(result, columns, expr, y_true, best_formulas, min_f1):
    f1 = fastmetrics.fast_f1_score(y_true, result)

    if min_f1 == -1 or f1 > min_f1:
        min_f1 = f1
        best_formulas.append((f1, columns, expr, result))

    """ BISECT VERSION """
    # if len(best_formulas) < 1000:
    #     f1 = f1_score(y_true, result)
    #     best_formulas.append((f1, columns, expr, result))
    # elif min_f1 == -1:
    #     f1 = f1_score(y_true, result)
    #     best_formulas.append((f1, columns, expr, result))
    #     best_formulas = sorted(best_formulas, key=lambda tup: tup[0], reverse=True)
    #     best_formulas.pop()
    #     min_f1 = best_formulas[-1][0]
    # else:
    #     f1 = f1_score(y_true, result)
    #     if f1 > min_f1:
    #         idx = bisect_left(best_formulas, f1, key=lambda tup: tup[0])
    #         best_formulas.insert(idx,  (f1, columns, expr, result))
    #         # for idx in range(len(best_formulas)):
    #         #     if f1 > best_formulas[idx][0]:
    #         #         break
    #         # best_formulas.insert(idx, (f1, columns, expr, result))
    #         best_formulas.pop()
    #         min_f1 = best_formulas[-1][0]

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
    models = tupleList_to_df(best_formulas)
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


def add_metrics(best_formulas, y_true):
    for i in range(len(best_formulas)):
        metrics = compute_metrics(y_true, best_formulas[i][-1])
        best_formulas[i] = metrics + best_formulas[i]
    return best_formulas