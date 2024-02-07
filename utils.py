import pandas as pd
from itertools import product, combinations
import os.path
import copy
from math import factorial
import operator
# import re
import numpy as np
from numba import njit

from metrics_utils import compare_model_similarity
# from viztracer import log_sparse


def apply_bool_func(features, bool_func):
    n_features = features.shape[1]
    indices = np.dot(features, 2**np.arange(n_features)[::-1])
    return bool_func[indices]


def get_1var_importance_order(best_models, subset_size, columns_number, parent_features_dict):
    columns_set = set()
    columns_ordered = []
    #reverse parent_features_dict for easier parent access
    feature_parent_dict = {}
    for key in parent_features_dict.keys():
        for val in parent_features_dict[key]:
            feature_parent_dict[val]=key
    
    parent_features_ordered = {}
    for i in range(len(best_models)):
        column_names = best_models[i]['columns']
        for column_name in column_names:
            parent = feature_parent_dict[column_name]
            if column_name not in columns_set:
                columns_set.add(column_name)
                columns_ordered.append(column_name)
                if parent in parent_features_ordered.keys():
                    parent_features_ordered[parent].append(column_name)
                else:
                    parent_features_ordered[parent] = [column_name]
    if subset_size == 1 and len(columns_ordered) != columns_number:
        print('Something went WRONG with feature importance: len(columns_ordered) != columns_number')
    return columns_ordered, parent_features_ordered

# P.S. Считать фича импотанс считать до фильтровки моделей. Эта функция возвращает упорядоченный датафрейм с важностью, 
# чтобы получить поселодовательность, надо взять list(importances_df.index)
def get_feature_importance(best_models, subset_size, project_name):
    feature_scores = {}
    for i in range(len(best_models)):
        for col in best_models[i]['columns']:
            if col not in feature_scores.keys():
                feature_scores[col] = 1
            else:
                feature_scores[col] += 1
    sorted_importance_dict = {k: v for k, v in sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)}
    importances_df = pd.DataFrame.from_dict(sorted_importance_dict, orient='index', columns = ['importance'])
    importances_df.to_pickle(f'./FeatureImportances/{project_name}_subset_size_{subset_size}.pkl')
    return importances_df

# In case if the number of features reduces with increase of subset_size, 
# just add features from the old list to the end of the new list
# нужно запускать если subset_size > 1
def add_missing_features(sorted_features_old, sorted_features_new):
    difference = set(sorted_features_old) - set(sorted_features_new)
    for item in difference:
        sorted_features_new.append(item)
    return sorted_features_new


# @log_sparse
def similarity_filtering(best_models, metric, min_jac_score, min_same_parents):
    filtered_models = []

    for i, model in enumerate(best_models):
        is_duplicate = False

        for j, other_model in enumerate(filtered_models):
            if compare_model_similarity(model, other_model, metric, min_jac_score, min_same_parents):
                if model["f1_1"] == other_model["f1_1"]:
                    if model["number_of_binary_operators"] < other_model["number_of_binary_operators"]:
                        filtered_models[j] = model  # Replace the duplicate with the current model
                is_duplicate = True
                break

        if not is_duplicate:
            filtered_models.append(model)

    return filtered_models


# List of tuples (best_models) to dataframe
# @log_sparse
def list_to_df(best_formulas):
    models = pd.DataFrame.from_records(best_formulas)
    return models


# Generator of boolean formulas in str format using truth tables
#def model_string_gen(vars_num):
#    inputs = list(product([False, True], repeat=vars_num))
#    for output in product([False, True], repeat=len(inputs)):
#        terms = []
#        for j in range(len(output)):
#            if output[j]:
#                terms.append(' & '.join(['df_np_cols[' + str(i) +']' if input_ else '~df_np_cols[' + str(i) +']' for i, input_ in enumerate(inputs[j])]))
#        if not terms:
#            terms = ['False']
#            continue
#        expr = ' | '.join(terms)
#        yield expr

def model_string_gen(vars_num, variables):
    inputs = list(product([False, True], repeat=vars_num))
    list_of_outputs = []
    #print(inputs)
    for output in product([False, True], repeat=len(inputs)):
        if tuple(map(operator.not_, output)) not in list_of_outputs:
            list_of_outputs.append(output)
            terms = []
            for j in range(len(output)):
                if output[j]:
                    terms.append(' & '.join([variables[i] if input_ else '~' + variables[i] for i, input_ in enumerate(inputs[j])]))
            if not terms:
                terms = ['False']
                continue
            expr = ' | '.join(terms)
            yield expr, output


def outputs_to_model_string(output, vars_num, variables):
    terms = []
    inputs = list(product([False, True], repeat=vars_num))
    for j in range(len(output)):
        if output[j]:
            terms.append(' & '.join([variables[i] if input_ else '~' + variables[i] for i, input_ in enumerate(inputs[j])]))
    if not terms:
        terms = ['False']
    expr = ' | '.join(terms)
    return expr


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


def get_1var_importance_config(main_config, columns_number):
    config_1_variable = copy.deepcopy(main_config)
    config_1_variable['rules_generation_params']['subset_size'] = 1
    config_1_variable['rules_generation_params']['process_number'] = min(2, main_config['rules_generation_params']['process_number'])
    subset_number = factorial(columns_number) / factorial(columns_number - config_1_variable['rules_generation_params']['subset_size'])
    config_1_variable['rules_generation_params']['batch_size'] = int(subset_number // 2) + 1
    config_1_variable['rules_generation_params']['crop_number'] = columns_number * 2
    config_1_variable['rules_generation_params']['crop_number_in_workers'] = None
    config_1_variable['similarity_filtering_params']['sim_metric'] = 'PARENT'
    config_1_variable['similarity_filtering_params']['min_same_parents'] = 2
    
    return config_1_variable


# Simpify expression with boolean.py library
# In order to simplify and find sums in expression we need to
# replace 'df[columns[{i}]]' with one-character variables
def simplify_expr(expr, subset_size, variables, algebra):
    simple_expr = str(algebra.parse(expr).simplify())
    return simple_expr


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


def add_readable_simple_formulas(best_models, subset_size):
    for j in range(len(best_models)):
        best_models[j]['simple_formula'] = best_models[j]['expr']
        for i in range(subset_size):
            best_models[j]['simple_formula'] = best_models[j]['simple_formula'].replace(f'df_np_cols[{i}]', best_models[j]['columns'][i])
    return best_models


def post_simplify(row, subset_size, variables, algebra):
    if row['is_negated']:
        simple_formula = simplify_expr(row['expr'], subset_size, variables, algebra)
        for i in range(subset_size):
            simple_formula = simple_formula.replace(variables[i], f'col{i}')
        for i in range(subset_size):
            simple_formula = simple_formula.replace(f'col{i}', row['columns'][i])
        return simple_formula
    else:
        return row['simple_formula']


# @log_sparse
def beautify_simple(df):
    df['readable_formula'] = df.apply(lambda x: x['readable_formula'].replace('~', ' ~'), axis=1)
    df['readable_formula'] = df.apply(lambda x: x['readable_formula'].replace('&', ' & '), axis=1)
    df['readable_formula'] = df.apply(lambda x: x['readable_formula'].replace('|', ' | '), axis=1)


# @log_sparse
def beautify_summed(df, subset_size, variables):
    df['summed_expr'] = df.apply(lambda x: x['summed_expr'].replace('\'', '') if x['summed_expr'] is not None else None, axis=1)
    df['summed_expr'] = df.apply(lambda x: x['summed_expr'].replace('{', '').replace('}', '') if x['summed_expr'] is not None else None, axis=1)
    for i in range(len(df['columns'][0])):
        df['summed_expr'] = df.apply(lambda x: x['summed_expr'].replace(variables[i], f' {variables[i]} ') if x['summed_expr'] is not None else None, axis=1)
    for i in range(len(df['columns'][0])):
        df['summed_expr'] = df.apply(lambda x: x['summed_expr'].replace(f' {variables[i]} ', x['columns'][i]) if x['summed_expr'] is not None else None, axis=1)


def log_exec(file_name, rows_num, cols_num, elapsed_time, sim_metric, min_jac_score, min_same_parents, quality_metric, subset_size, \
            process_number, batch_size, crop_features, crop_number, crop_number_in_workers, excessive_models_num_coef, \
            filter_similar_between_reloads, dataset_frac, incremental_run, crop_features_after_size):
    if os.path.exists("./Output/log.xlsx"):
        log = pd.read_excel('./Output/log.xlsx')
        search_idx = log.loc[(log['dataset'] == file_name) & (log['sim_metric'] == sim_metric) & \
            (log['subset_size'] == subset_size) & (log['rows_num'] == rows_num) & (log['cols_num'] == cols_num) & \
            (log['process_number'] == process_number) & (log['batch_size'] == batch_size) & \
            (log['dataset_frac'] == dataset_frac)].index.tolist()
        if len(search_idx) == 1:
            log.loc[search_idx, ['elapsed_time']] = elapsed_time
            with pd.ExcelWriter('./Output/log.xlsx', mode="w", engine="openpyxl") as writer:
                log.to_excel(writer, sheet_name='Logs', index=False, freeze_panes=(1,1))
        else:
            new_row = pd.DataFrame(data={'dataset': [file_name], 'dataset_frac': [dataset_frac], 'rows_num': [rows_num], 'cols_num': [cols_num], 'sim_metric': [sim_metric], \
                'subset_size': [subset_size], 'batch_size': [batch_size], 'process_number': [process_number], 'elapsed_time': [elapsed_time]})
            log = pd.concat([log, new_row], ignore_index=True)
            with pd.ExcelWriter('./Output/log.xlsx', mode="w", engine="openpyxl") as writer:
                log.to_excel(writer, sheet_name='Logs', index=False, freeze_panes=(1,1))
    else:
        log = pd.DataFrame(data={'dataset': [file_name], 'dataset_frac': [dataset_frac], 'rows_num': [rows_num], 'cols_num': [cols_num], 'sim_metric': [sim_metric], \
             'subset_size': [subset_size], 'batch_size': [batch_size], 'process_number': [process_number], 'elapsed_time': [elapsed_time]})
        with pd.ExcelWriter('./Output/log.xlsx', mode="w", engine="openpyxl") as writer:
                log.to_excel(writer, sheet_name='Logs', index=False, freeze_panes=(1,1))


def get_readable_size(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            # return "%3.1f %s%s" % (num, unit, suffix)
            return f"{num:.1f} {unit}{suffix}"
        num /= 1024.0
    # return "%.1f %s%s" % (num, 'Yi', suffix)
    return f"{num:.1f} Yi{suffix}"