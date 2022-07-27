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


def log_exec(execution_type, subset_size, elapsed_time, process_number, batch_size):
    if os.path.exists("./Output/log.xlsx"):
        log = pd.read_excel('./Output/log.xlsx')
        search_idx = log.loc[(log['execution_type'] == execution_type) & (log['subset_size'] == subset_size) \
            & (log['process_number'] == process_number) & (log['batch_size'] == batch_size)].index.tolist()
        if len(search_idx) == 1:
            log.loc[search_idx, ['elapsed_time']] = elapsed_time
            with pd.ExcelWriter('./Output/log.xlsx', mode="w", engine="openpyxl") as writer:
                log.to_excel(writer, sheet_name='Logs', index=False, freeze_panes=(1,1))
        else:
            new_row = pd.DataFrame(data={'execution_type': [execution_type], 'subset_size': [subset_size], 'elapsed_time': [elapsed_time], \
                'process_number': [process_number], 'batch_size': [batch_size]})
            log = pd.concat([log, new_row], ignore_index=True)
            with pd.ExcelWriter('./Output/log.xlsx', mode="w", engine="openpyxl") as writer:
                log.to_excel(writer, sheet_name='Logs', index=False, freeze_panes=(1,1))
    else:
        log = pd.DataFrame(data={'execution_type': [execution_type], 'subset_size': [subset_size], 'elapsed_time': [elapsed_time], \
            'process_number': [process_number], 'batch_size': [batch_size]})
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