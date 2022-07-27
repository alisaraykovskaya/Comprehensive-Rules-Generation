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

# import bisect
from itertools import product, combinations, permutations
import time


def find_best_model_sequential(X_train, y_train, X_test, y_test, subset_size, file_name):
    best_formulas = find_best_model(X_train, y_train, subset_size)
    best_formulas = sorted(best_formulas, key=lambda tup: tup[0], reverse=True)
    best_formulas = add_metrics(best_formulas, y_train)
    models = tupleList_to_df(best_formulas)
    print('Best model validation results:')
    validate_result = validate_model(models['columns'][0], models['expr'][0], X_test, y_test)
    print(validate_result)
    get_formulas(models)
    beautify_formulas(models)
    models.drop('result', axis=1, inplace=True)
    models.to_excel(f"BestModels_{file_name}.xlsx", index=False, freeze_panes=(1,1))


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


# Substitute "columns[column_name]" with just "column_name" in formulas
def get_formulas(df):
    df['formula'] = df.apply(lambda x: x['expr'].replace(f'columns[0]', x['columns'][0]), axis=1)
    for i in range(1, len(df['columns'][0])):
        df['formula'] = df.apply(lambda x: x['formula'].replace(f'columns[{i}]', x['columns'][i]), axis=1)
    # df.drop('expr', axis=1, inplace=True)


# List of tuples (best_models) to dataframe
def tupleList_to_df(best_formulas, in_training=False, parallel=False):
    if in_training:
        models = pd.DataFrame(data=best_formulas, columns=['f1', 'columns', 'expr', 'result'])
    elif parallel:
        models = pd.DataFrame(data=best_formulas, columns=['f1_1', 'f1_0', 'tn', 'fp', 'fn', 'tp', 'precision_1', 'precision_0', 'recall_1', 'recall_0', 'rocauc', 'accuracy', 'elapsed_time', 'summed_expr', 'columns', 'expr', 'simple_formula'])
    else:
        models = pd.DataFrame(data=best_formulas, columns=['precision_0', 'recall_0', 'precision_1', 'recall_1', 'rocauc', 'accuracy', 'f1', 'columns', 'expr', 'result'])
    return models


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
        return (precision_0, recall_0, precision_1, recall_1, rocauc, accuracy, f1)
    return (precision_0, recall_0, precision_1, recall_1, rocauc, accuracy)


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


def check_model_perfomance(result, columns, expr, y_true, best_formulas, min_f1):
    if len(best_formulas) < 3000:
        f1 = f1_score(y_true, result)
        if min_f1 == -1 or f1 > min_f1:
            best_formulas.append((f1, columns, expr, result))
    else:
        f1 = f1_score(y_true, result)
        best_formulas.append((f1, columns, expr, result))
        best_formulas = sorted(best_formulas, key=lambda tup: tup[0], reverse=True)
        best_formulas = best_formulas[:1000]
        models = tupleList_to_df(best_formulas, in_training=True)
        models.drop('result', axis=1, inplace=True)
        get_formulas(models)
        beautify_formulas(models)
        models.to_excel("models_tmp.xlsx", index=False, freeze_panes=(1,1))
        min_f1 = best_formulas[-1][0]

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


def find_best_model(df, y_true, subset_size, only_best=False):
    formulas_number = 2**(2**subset_size)
    best_formulas = []
    min_f1 = -1
    elapsed_time = 0
    start = time.time()
    for expr in tqdm(model_string_gen(subset_size), total=formulas_number-1):
        for columns in combinations(df.columns, subset_size):
            columns_lst = list(columns)
            model = eval('lambda df, columns: ' + expr)
            # print(~df.isnull().any(axis=1))
            tmp_df = df[~df.isnull().any(axis=1)]
            y_true_tmp = y_true[~df.isnull().any(axis=1)]
            result = model(tmp_df, columns).to_numpy()

            best_formulas, min_f1 = check_model_perfomance(result, columns, expr, y_true_tmp, best_formulas, min_f1)
    elapsed_time = time.time() - start
    print(f'Elapsed time {elapsed_time} seconds')
    return best_formulas


######################################################################################################################


def add_metrics(best_formulas, y_true):
    for i in range(len(best_formulas)):
        metrics = compute_metrics(y_true, best_formulas[i][-1])
        best_formulas[i] = metrics + best_formulas[i]
    return best_formulas


def validate_model(columns, expr, X_test, y_test):
    model = eval('lambda df, columns: ' + expr)
    result = model(X_test, columns)
    metrics = [compute_metrics(y_test, result, compute_f1=True)]
    metrics = pd.DataFrame(data=metrics, columns=['precision_0', 'recall_0', 'precision_1', 'recall_1', 'rocauc', 'accuracy', 'f1'])
    return metrics