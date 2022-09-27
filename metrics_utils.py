import numpy as np
import pandas as pd
from numba import vectorize
import re


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


@vectorize
def generate_intersection(x, y):
    """
    NumPy ufunc implemented with Numba that generates the intersection of two arrays.
    Serves as a helper method for fast_jaccard_score().
    """
    if x & y:
        return 1

    else:
        return 0


@vectorize
def generate_union(x, y):
    """
    NumPy ufunc implemented with Numba that generates the union of two arrays.
    Serves as a helper method for fast_jaccard_score().
    """
    if x | y:
        return 1

    else:
        return 0


def fast_jaccard_score(y_true, y_pred):
    intersection = generate_intersection(y_true, y_pred)
    union = generate_union(y_true, y_pred)
    numer = np.count_nonzero(intersection == 1)
    denom = np.count_nonzero(union == 1)
    return 0 if denom == 0 else numer / denom


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
def compare_model_similarity(model_dict1, model_dict2, metric, min_jac_score=0.9, min_same_parents=1):
    res = False
    if metric == "JAC_SCORE":
        res = fast_jaccard_score(model_dict1['result'], model_dict2['result']) >= min_jac_score
    elif metric == 'PARENT':
        res = parent_features_similarity(model_dict1['columns_set'], model_dict2['columns_set'], min_same_parents)
    if not res and model_dict1['simple_formula'] == model_dict2['simple_formula']:
        return True
    return res


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


