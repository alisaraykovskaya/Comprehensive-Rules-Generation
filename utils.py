import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os.path
import fastmetrics


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
def compare_model_similarity(model1, model2, columns1, columns2, metric, min_jac_score=.9):
    if metric == "JAC_SCORE":
        return fastmetrics.fast_jaccard_score(model1, model2) >= min_jac_score
    elif metric == 'PARENT(list)':
        parent_set1 = get_parent_set(columns1)
        parent_set2 = get_parent_set(columns2)
        return parent_features_similarity(parent_set1, parent_set2)
    elif metric == 'PARENT(set)':
        # print(columns1, columns2)
        return parent_features_similarity(columns1, columns2)


def log_exec(file_name, execution_type, sim_metric, subset_size, elapsed_time, process_number, batch_size):
    if os.path.exists("./Output/log.xlsx"):
        log = pd.read_excel('./Output/log.xlsx')
        search_idx = log.loc[(log['dataset'] == file_name) & (log['execution_type'] == execution_type) & (log['sim_metric'] == sim_metric) & \
            (log['subset_size'] == subset_size) & (log['process_number'] == process_number) & (log['batch_size'] == batch_size)].index.tolist()
        if len(search_idx) == 1:
            log.loc[search_idx, ['elapsed_time']] = elapsed_time
            with pd.ExcelWriter('./Output/log.xlsx', mode="w", engine="openpyxl") as writer:
                log.to_excel(writer, sheet_name='Logs', index=False, freeze_panes=(1,1))
        else:
            new_row = pd.DataFrame(data={'dataset': [file_name], 'execution_type': [execution_type], 'sim_metric': [sim_metric], \
                'subset_size': [subset_size], 'elapsed_time': [elapsed_time], 'process_number': [process_number], 'batch_size': [batch_size]})
            log = pd.concat([log, new_row], ignore_index=True)
            with pd.ExcelWriter('./Output/log.xlsx', mode="w", engine="openpyxl") as writer:
                log.to_excel(writer, sheet_name='Logs', index=False, freeze_panes=(1,1))
    else:
        log = pd.DataFrame(data={'dataset': [file_name], 'execution_type': [execution_type], 'sim_metric': [sim_metric], \
            'subset_size': [subset_size], 'elapsed_time': [elapsed_time], 'process_number': [process_number], 'batch_size': [batch_size]})
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
        return (precision_0, recall_0, precision_1, recall_1, rocauc, accuracy, f1)
    return (precision_0, recall_0, precision_1, recall_1, rocauc, accuracy)


def validate_model(columns, expr, X_test, y_test):
    model = eval('lambda df, columns: ' + expr)
    result = model(X_test, columns)
    metrics = [compute_metrics(y_test, result, compute_f1=True)]
    metrics = pd.DataFrame(data=metrics, columns=['precision_0', 'recall_0', 'precision_1', 'recall_1', 'rocauc', 'accuracy', 'f1'])
    return metrics