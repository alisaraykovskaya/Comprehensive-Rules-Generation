from os import path, mkdir
import platform
from time import time
from math import factorial
from multiprocessing import freeze_support, cpu_count, Pool
from itertools import combinations, chain, islice
from copy import deepcopy
import operator
from datetime import datetime
import itertools
# import psutil
# from memory_profiler import profile
# import sys
# import gc
# import numexpr as ne

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, classification_report

import boolean
# from numba import njit

from utils import model_string_gen, simplify_expr, find_sum, sums_generator, similarity_filtering, list_to_df, beautify_simple, beautify_summed
from utils import get_1var_importance_order, get_feature_importance, add_missing_features, outputs_to_model_string
from metrics_utils import count_actual_subset_size, count_operators, count_vars, count_confusion_matrix, get_parent_set
from metrics_utils import calculate_metrics_from_conf_matrix
from utils import make_nan_mask, apply_nan_mask_list
from utils import apply_bool_func
# from itertools import product

# import cProfile
# import pstats


class CRG:
    def __init__(
        self,
        binarizer,
        project_name='',
        load_from_pkl=True,
        subset_size=3,
        quality_metric="f1",
        process_number="default",
        batch_size=10000,
        filter_similar_between_reloads=True,
        crop_number=1000,
        crop_number_in_workers=False,
        excessive_models_num_coef=3,
        dataset_frac=1,
        crop_features=-1,
        crop_parent_features=20,
        complexity_restr_operators=None,
        complexity_restr_vars=None,
        restrict_complexity_after_size=1,
        time_restriction_seconds=None,
        incremental_run=True,
        crop_features_after_size=1,
        sim_metric="PARENT",
        min_jac_score=0.9,
        min_same_parents=2
    ):
        self.subset_size = subset_size
        self.binarizer = binarizer
        self.project_name = project_name
        self.load_from_pkl = load_from_pkl
        self.quality_metric = quality_metric
        self.process_number = process_number
        self.batch_size = batch_size
        self.filter_similar_between_reloads = filter_similar_between_reloads
        self.crop_number = crop_number
        self.crop_number_in_workers = crop_number_in_workers
        self.excessive_models_num_coef = excessive_models_num_coef
        self.dataset_frac = dataset_frac
        self.crop_features = crop_features
        self.crop_parent_features = crop_parent_features
        self.complexity_restr_operators=complexity_restr_operators
        self.complexity_restr_vars=complexity_restr_vars
        self.restrict_complexity_after_size=restrict_complexity_after_size
        self.time_restriction_seconds = time_restriction_seconds
        self.incremental_run = incremental_run
        self.crop_features_after_size = crop_features_after_size
        self.sim_metric = sim_metric
        self.min_jac_score = min_jac_score
        self.min_same_parents = min_same_parents
        

        if 'windows' in platform.system().lower():
            freeze_support()
        if not path.exists('Output'):
            mkdir('Output')
        if not path.exists('FeatureImportances'):
            mkdir('FeatureImportances')
        if self.process_number=='default':
            self.process_number = int(max(cpu_count()*.9, 1))

        self.best_models_dict = {}
        self.best_models = []
        self.raw_df = None
        self.df = None
        self.y_true = None
        self.df_dict = {}
        self.all_formulas = []
        self.start_time = -1
        self.columns_ordered = []

        self.onevar_crop_number = None
        self.onevar_crop_number_in_workers = None
        self.onevar_sim_metric = "PARENT"
        self.onevar_min_same_parents = 2

        # process = psutil.Process()
        # memory_info = process.memory_info()
        # memory_usage = memory_info.rss / 1024 / 1024
        # print(f"Memory usage: {memory_usage:.2f} MB")


    def fit(self, df, y_true):
        self.raw_df = df.copy()
        self.raw_df['Target'] = y_true

        print('BINARIZING DATA...')
        self.df = self.binarizer.fit_transform(self.raw_df)
        # if self.project_name == "DivideBy30":
        #     self.df = self.df[['Div_By_15=True', 'Div_By_10=True', 'Div_By_6=True', 'Target']]
        self.parent_features_dict = self.binarizer.parent_features_dict

        # using only fraction of dataset
        if self.dataset_frac != 1:
            self.df, _ = train_test_split(self.df, train_size=self.dataset_frac, stratify=self.df['Target'], random_state=12)
        self.y_true = self.df['Target']
        self.df = self.df.drop(columns=['Target'])
        self.columns_ordered = self.df.columns
        columns_number = len(self.columns_ordered)

        print(f'\nproject_name: {self.project_name}  columns_number: {columns_number}  observations_number: {self.df.shape[0]}')

        print('\nRUNNING ON subset_size=1 TO DETERMINE INITIAL FEATURE IMPORTANCES')
        start_time = time()
        best_1_variable = self.find_best_models(is_onevar=True)
        elapsed_time = time() - start_time
        self.log_exec(subset_size=1, elapsed_time=elapsed_time, rows_num=self.df.shape[0], cols_num=self.df.shape[1])
        self.columns_ordered, parent_features_ordered = get_1var_importance_order(best_1_variable, subset_size=1, columns_number=columns_number, parent_features_dict = self.parent_features_dict)
        
        # Crop features with respect to parents
        for parent in parent_features_ordered.keys():
            parent_features_ordered[parent] = parent_features_ordered[parent][:self.crop_parent_features]
        features_to_use = list(itertools.chain(*list(parent_features_ordered.values())))
        columns_ordered = []
        for col in self.columns_ordered:
            if col in features_to_use:
                columns_ordered.append(col)
        self.columns_ordered = columns_ordered
        if self.crop_features != -1 and (self.incremental_run and self.crop_features_after_size == 1 or not self.incremental_run):
            self.columns_ordered = self.columns_ordered[:self.crop_features]
            print(f'\nNumber of features is croped by {self.crop_features}')
        self.df = self.df[self.columns_ordered]
        print(f'\nTop 5 important features: {self.columns_ordered[:5]}')

        if not self.incremental_run:
            print('\nBEGIN TRAINING...')
            start_time = time()
            best_models = self.find_best_models(is_onevar=False)    
            elapsed_time = time() - start_time
            self.log_exec(subset_size=self.subset_size, elapsed_time=elapsed_time, rows_num=self.df.shape[0], cols_num=self.df.shape[1])
        else:
            main_subset_size = self.subset_size
            for i in range(2, main_subset_size + 1):
                print(f'\nBEGIN TRAINING FOR subset_size={i}...')
                self.subset_size = i
                start_time = time()
                best_models = self.find_best_models(is_onevar=False)    
                elapsed_time = time() - start_time
                self.log_exec(subset_size=self.subset_size, elapsed_time=elapsed_time, rows_num=self.df.shape[0], cols_num=self.df.shape[1])
                
                columns_ordered_new = get_feature_importance(best_models, self.subset_size, self.project_name)
                columns_ordered_new = list(columns_ordered_new.index)
                self.columns_ordered = add_missing_features(self.columns_ordered, columns_ordered_new)
                if self.crop_features != -1 and self.crop_features_after_size == i:
                    self.columns_ordered = self.columns_ordered[:self.crop_features]
                    print(f'\nNumber of features is croped by {self.crop_features}')
                self.df = self.df[self.columns_ordered]


    def predict(self, raw_df_test, y_test, subset_size=1, incremental=False):
        print('BINARIZING TEST DATA...')
        df_test = self.binarizer.transform(raw_df_test)
        df_test_dict = {}
        for col in df_test.columns:
            df_test_dict[col] = {'data': df_test[col].values.astype(bool), 'nan_mask': pd.isna(df_test[col]).values}
        result = []
        variables = list(map(chr, range(122, 122-subset_size,-1)))
        test_df_len = len(df_test)
        for current_size in range(1, subset_size+1):
            models = self.best_models_dict[current_size]
            for model_info in models:
                columns = model_info['columns']
                nan_mask = np.full_like(self.y_true, True, dtype=bool)
                features_subset = []
                df_nan_cols = []
                for col in columns:
                    features_subset.append(df_test_dict[col]['data'])
                    df_nan_cols.append(df_test_dict[col]['nan_mask'])
                features_subset = np.column_stack(features_subset)
                df_nan_cols = np.array(df_nan_cols)
                nan_mask = make_nan_mask(nan_mask, df_nan_cols, subset_size)
                bool_func = model_info['bool_func']
                result = apply_bool_func(features_subset, bool_func)
                result = apply_nan_mask_list(result, nan_mask)
                tp, fp, fn, tn = count_confusion_matrix(y_test.values, result)
                precision, recall, f1, rocauc, accuracy = calculate_metrics_from_conf_matrix(tp, fp, fn, tn)
                precision_0 = 0 if (tn + fn) == 0 else tn / (tn + fn)
                recall_0 = 0 if (tn + fp) == 0 else tn / (tn + fp)
                f1_0 = 0 if (precision_0 + recall_0) == 0 else 2 * (precision_0 * recall_0) / (precision_0 + recall_0)

                model_info["test_f1_1"] = f1
                model_info["test_f1_0"] = f1_0
                model_info["test_precision_1"] = precision
                model_info["test_precision_0"] = precision_0
                model_info["test_recall_1"] = recall
                model_info["test_recall_0"] = recall_0
                model_info["test_rocauc"] = rocauc
                model_info["test_accuracy"] = accuracy

            if len(models) > 0:
                models_to_excel = list_to_df(models)
                models_to_excel.drop(['columns_set', 'result'], axis=1, inplace=True)
                beautify_simple(models_to_excel)
                beautify_summed(models_to_excel, current_size, variables)

                models_to_excel = models_to_excel[['tn', 'fp', 'fn', 'tp', 'precision_1', 'recall_1', 'rocauc', 'f1_1', 'nan_ratio', 'accuracy', \
                    'test_f1_1', 'test_precision_1', 'test_recall_1', 'test_rocauc', 'test_accuracy', 'elapsed_time', 'columns', 'summed_expr', \
                    'readable_formula', 'number_of_binary_operators', 'max_freq_of_variables']]
                models_to_excel.rename(columns={'precision_1': 'precision', 'recall_1': 'recall', 'f1_1': 'f1', 'test_precision_1': 'test_precision', \
                                                'test_recall_1': 'test_recall', 'test_f1_1': 'test_f1'}, inplace=True)

                with pd.ExcelWriter(f"./Output/{self.project_name}_BestModels.xlsx", mode="a", if_sheet_exists='replace', engine="openpyxl") as writer:
                    models_to_excel.to_excel(writer, sheet_name=f'Size {current_size}', index=False, freeze_panes=(1,1))


    
    # def ensemble_predict(self, raw_df_test, y_test=None, subset_size=1, k_best=1, incremental=False):
    #     print('BINARIZING TEST DATA...')
    #     df_test = self.binarizer.transform(raw_df_test)
    #     df_test_dict = {}
    #     for col in df_test.columns:
    #         df_test_dict[col] = {'data': df_test[col].values.astype(bool), 'nan_mask': pd.isna(df_test[col]).values}
    #     result = []
    #     variables = list(map(chr, range(122, 122-subset_size,-1)))
    #     test_df_len = len(df_test)
    #     if not incremental:
    #         if isinstance(k_best, int):
    #             if subset_size not in self.best_models_dict:
    #                 print(f"The algorithm did not run for subset_size={subset_size}")
    #                 return None
    #             models = self.best_models_dict[subset_size][:k_best]
    #             for model_dict in models:
    #                 columns = model_dict['columns']
    #                 nan_mask = np.full_like(df_test.shape[0], True, dtype=bool)
    #                 df_np_cols = []
    #                 df_nan_cols = []
    #                 for col in columns:
    #                     df_np_cols.append(df_test_dict[col]['data'])
    #                     df_nan_cols.append(df_test_dict[col]['nan_mask'])
    #                 df_np_cols = np.array(df_np_cols)
    #                 df_nan_cols = np.array(df_nan_cols)
    #                 local_dict = {}
    #                 for i in range(len(variables)):
    #                     local_dict[variables[i]] = df_np_cols[i]
    #                 expr = model_dict['expr']
    #                 # tmp = ne.evaluate(expr, local_dict=local_dict)
    #                 tmp = custom_eval(expr, local_dict, test_df_len)
    #                 # tmp = custom_eval2(expr, local_dict, test_df_len)
    #                 result.append(tmp)
    #             result = np.stack(result, axis=-1)
    #             print(result.shape)
    #             result = np.rint(np.mean(result, axis=1)).astype(bool)
    #         return result
    #     else:
    #         predictions_df = {
    #             "subset_size": [],
    #             "ensemble_size": [],
    #             "f1_1": [],
    #             "f1_0": [],
    #             "precision_1": [],
    #             "precision_0": [],
    #             "recall_1": [],
    #             "recall_0": [],
    #             "rocauc": [],
    #             "accuracy": [],
    #         }
    #         for curent_size in range(1, subset_size+1):
    #             for ensemble_size in range(1, k_best+1):
    #                 result = []
    #                 print(f"SUBSET_SIZE: {curent_size}, ENSEMBLE_SIZE: {ensemble_size}")
    #                 models = self.best_models_dict[curent_size][:ensemble_size]
    #                 for model_dict in models:
    #                     columns = model_dict['columns']
    #                     nan_mask = np.full_like(df_test.iloc[:, 0], True, dtype=bool)
    #                     local_dict = {}
    #                     df_nan_cols = []
    #                     for i in range(curent_size):
    #                         local_dict[variables[i]] = df_test_dict[columns[i]]['data']
    #                         df_nan_cols.append(df_test_dict[columns[i]]['nan_mask'])
    #                     df_nan_cols = np.array(df_nan_cols)
    #                     nan_mask = make_nan_mask(nan_mask, df_nan_cols, curent_size)
    #                     expr = model_dict['expr']
    #                     # tmp = ne.evaluate(expr, local_dict=local_dict)
    #                     tmp = custom_eval(expr, local_dict, test_df_len)
    #                     # tmp = custom_eval2(expr, local_dict, test_df_len)
    #                     result.append(tmp)
    #                 result = np.stack(result, axis=-1)
    #                 result = np.rint(np.mean(result, axis=1)).astype(float)
    #                 result = apply_nan_mask_list(result, nan_mask)
    #                 tp, fp, fn, tn = count_confusion_matrix(y_test.values, result)
    #                 precision, recall, f1, rocauc, accuracy, fpr = calculate_metrics_for(tp, fp, fn, tn)
    #                 precision_0 = 0 if (tn + fn) == 0 else tn / (tn + fn)
    #                 recall_0 = 0 if (tn + fp) == 0 else tn / (tn + fp)
    #                 f1_0 = 0 if (precision_0 + recall_0) == 0 else 2 * (precision_0 * recall_0) / (precision_0 + recall_0)

    #                 predictions_df["subset_size"].append(curent_size)
    #                 predictions_df["ensemble_size"].append(ensemble_size)
    #                 predictions_df["f1_1"].append(f1)
    #                 predictions_df["f1_0"].append(f1_0)
    #                 predictions_df["precision_1"].append(precision)
    #                 predictions_df["precision_0"].append(precision_0)
    #                 predictions_df["recall_1"].append(recall)
    #                 predictions_df["recall_0"].append(recall_0)
    #                 predictions_df["rocauc"].append(rocauc)
    #                 predictions_df["accuracy"].append(accuracy)
    #                 # print(classification_report(y_test, result))
    #                 print(f"f1_1: {f1}  f1_0: {f1_0}")
    #                 print(f"precision_1: {precision}  precision_0: {precision_0}")
    #                 print(f"recall_1: {recall}  recall_0: {recall_0}")
    #                 print(f"rocauc: {rocauc}  accuracy: {accuracy}")
    #                 print()
    #         predictions_df = pd.DataFrame(predictions_df)
    #         predictions_df.to_excel(f"./Output/BestModels_{self.project_name}_test.xlsx", index=False, freeze_panes=(1,1))


    def get_best_bool_func(self, features):
        n_features = features.shape[1]
        n_combinations = 2 ** n_features
        bool_func = np.zeros(n_combinations, dtype=bool)
        indices = np.dot(features, 2**np.arange(n_features)[::-1])

        pos_count_by_row = np.zeros(n_combinations)
        neg_count_by_row = np.zeros(n_combinations)

        for i in range(n_combinations):
            mask = indices == i
            pos_count_by_row[i] = len(self.y_true[mask & (self.y_true == True)])
            neg_count_by_row[i] = len(self.y_true[mask & (self.y_true == False)])

        if self.quality_metric == 'accuracy':
            bool_func[pos_count_by_row >= neg_count_by_row] = True
            return bool_func
        elif self.quality_metric == 'weighted_accuracy':
            total_pos = len(self.y_true[self.y_true == True])
            total_neg = len(self.y_true[self.y_true == False])
            bool_func[pos_count_by_row/total_pos >= neg_count_by_row/total_neg] = True
        else:
            formulas_len_with_neg = len(self.all_formulas) * 2
            formulas_len_no_neg = len(self.all_formulas)
            precision_list = np.zeros(formulas_len_with_neg)
            recall_list = np.zeros(formulas_len_with_neg)
            f1_list = np.zeros(formulas_len_with_neg)
            rocauc_list = np.zeros(formulas_len_with_neg)
            for i, formula_info in enumerate(self.all_formulas):
                formula_output = formula_info["output"]
                mask_pos = formula_output == True
                mask_neg = formula_output == False
                tp = np.sum(pos_count_by_row[mask_pos])
                # FP - это все positive counts, которые попадают под отрицательные outputs
                fp = np.sum(neg_count_by_row[mask_pos])
                # TN - это все negative counts, которые попадают под отрицательные outputs
                tn = np.sum(neg_count_by_row[mask_neg])
                # FP - это все positive counts, которые попадают под отрицательные outputs
                fn = np.sum(pos_count_by_row[mask_neg])
                precision, recall, f1, rocauc, accuracy = calculate_metrics_from_conf_matrix(tp, fp, fn, tn)
                fn_neg = tp
                tn_neg = fp
                fp_neg = tn
                tp_neg = fn
                precision_neg, recall_neg, f1_neg, rocauc_neg, accuracy_neg = calculate_metrics_from_conf_matrix(tp_neg, fp_neg, fn_neg, tn_neg)
                
                precision_list[i] = precision
                recall_list[i] = recall
                f1_list[i] = f1
                rocauc_list[i] = rocauc
                
                precision_list[i+formulas_len_no_neg] = precision_neg
                recall_list[i+formulas_len_no_neg] = recall_neg
                f1_list[i+formulas_len_no_neg] = f1_neg
                rocauc_list[i+formulas_len_no_neg] = rocauc_neg

            if self.quality_metric == 'f1_1':
                idx = np.argmax(f1_list)

            elif self.quality_metric == 'rocauc':
                idx = np.argmax(rocauc_list)

            elif self.quality_metric == 'recall_1':
                idx = np.argmax(recall_list)

            elif self.quality_metric == 'precision_1':
                idx = np.argmax(precision_list)

            else:
                raise Exception("Wrong quality metric")

            if idx < len(self.all_formulas):
                negated = False
            else:
                negated = True
                idx = idx - formulas_len_no_neg

            return self.all_formulas[idx], negated


    def worker_init(self):
        self.start_time = time()

    # @profile
    def worker(self, start_idx, end_idx, min_quality, subset_size, variables):
        if subset_size == 1:
            crop_number_in_workers = self.onevar_crop_number_in_workers
        else:
            crop_number_in_workers = self.crop_number_in_workers
        best_models = []
        algebra = boolean.BooleanAlgebra()
        for columns_tuple in islice(combinations(self.columns_ordered, subset_size), start_idx, end_idx):
            columns = list(columns_tuple)
            nan_mask = np.full_like(self.y_true, True, dtype=bool)
            features_subset = []
            df_nan_cols = []
            for i in range(subset_size):
                features_subset.append(self.df_dict[columns[i]]['data'])
                df_nan_cols.append(self.df_dict[columns[i]]['nan_mask'])
            features_subset = np.column_stack(features_subset)
            df_nan_cols = np.array(df_nan_cols)
            nan_mask = make_nan_mask(nan_mask, df_nan_cols, subset_size)
            nan_count = np.count_nonzero(nan_mask)
            formula_info, negated = self.get_best_bool_func(features_subset)

            if negated:
                output_key = "neg_output"
                simple_formula_key = "neg_simple_formula"
                summed_formula_key = "neg_summed_formula"
                readable_formula_key = "neg_readable_formula"
                number_of_binary_operators_key = "neg_number_of_binary_operators"
                max_freq_of_variables_key = "neg_max_freq_of_variables"
            else:
                output_key = "output"
                simple_formula_key = "simple_formula"
                summed_formula_key = "summed_formula"
                readable_formula_key = "readable_formula"
                number_of_binary_operators_key = "number_of_binary_operators"
                max_freq_of_variables_key = "max_freq_of_variables"

            result = apply_bool_func(features_subset, formula_info[output_key])
            result = apply_nan_mask_list(result, nan_mask)
            tp, fp, fn, tn = count_confusion_matrix(self.y_true, result)
            precision, recall, f1, rocauc, accuracy = calculate_metrics_from_conf_matrix(tp, fp, fn, tn)

            if self.quality_metric == 'f1_1':
                quality = f1
            elif self.quality_metric == 'precision_1':
                quality = precision
            elif self.quality_metric == 'recall_1':
                quality = recall
            elif self.quality_metric == 'rocauc':
                quality = rocauc
            elif self.quality_metric == 'accuracy' or self.quality_metric == 'weighted_accuracy':
                quality = accuracy

            if quality > min_quality:
                precision_0 = 0 if (tn + fn) == 0 else tn / (tn + fn)
                recall_0 = 0 if (tn + fp) == 0 else tn / (tn + fp)
                f1_0 = 0 if (precision_0 + recall_0) == 0 else 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
                columns_set = get_parent_set(columns)
                elapsed_time = time() - self.start_time
                readable_formula = formula_info[readable_formula_key]
                for i in range(subset_size):
                    readable_formula = readable_formula.replace(f"var[{i}]", columns[i])
                model_info = {'f1_1': f1, 'f1_0': f1_0, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'precision_1': precision, 'precision_0': precision_0,\
                    'recall_1': recall, 'recall_0': recall_0, 'rocauc': rocauc, 'accuracy': accuracy, 'elapsed_time': elapsed_time, 'bool_func': formula_info[output_key], \
                    'summed_expr': formula_info[summed_formula_key], 'columns': columns, 'expr': formula_info[simple_formula_key], 'expr_len': len(formula_info[simple_formula_key]), 'readable_formula': readable_formula, 'columns_set': columns_set, \
                    'number_of_binary_operators': formula_info[number_of_binary_operators_key], 'max_freq_of_variables': formula_info[max_freq_of_variables_key], 'nan_ratio': nan_count/self.y_true.shape[0],}
                if self.sim_metric == 'JAC_SCORE':
                    model_info['result'] = result
                else:
                    model_info['result'] = None
                best_models.append(model_info)
            if crop_number_in_workers is not None and len(best_models) >= crop_number_in_workers * self.excessive_models_num_coef:
                best_models.sort(key=lambda row: (row[self.quality_metric], -row['number_of_binary_operators']), reverse=True)
                best_models = best_models[:crop_number_in_workers]
                if len(self.best_models) > 0:
                    min_quality = best_models[-1][self.quality_metric]
        return best_models
    

    def is_formula_suitable(self, simple_expr, subset_size, actual_subset_size, number_of_binary_operators, max_freq_of_variables):
        if simple_expr == '1' or actual_subset_size < subset_size:
            return False
        if self.restrict_complexity_after_size < subset_size:
            if self.complexity_restr_operators is not None:
                if self.complexity_restr_operators == "subset_size+1":
                    complexity_restr_operators = subset_size + 1
                else:
                    complexity_restr_operators = self.complexity_restr_operators
                if number_of_binary_operators > complexity_restr_operators:
                    return False
            if self.complexity_restr_vars is not None and max_freq_of_variables > self.complexity_restr_vars:
                return False
            
        return True


    def find_best_models(self, is_onevar):
        excel_exist = False
        if path.exists(f"./Output/{self.project_name}_BestModels.xlsx"):
            excel_exist = True

        if is_onevar:
            subset_size = 1
            crop_number = self.onevar_crop_number
            sim_metric = self.onevar_sim_metric
            min_same_parents = self.onevar_min_same_parents
        elif self.min_same_parents == "subset_size":
            subset_size = self.subset_size
            crop_number = self.crop_number
            sim_metric = self.sim_metric
            min_same_parents = self.subset_size
        else:
            subset_size = self.subset_size
            crop_number = self.crop_number
            sim_metric = self.sim_metric
            min_same_parents = self.min_same_parents

        columns_number = len(self.columns_ordered)
        subset_number = int(factorial(columns_number) / factorial(columns_number - subset_size))

        batch_size = self.batch_size
        if batch_size > subset_number:
            batch_size = subset_number

        if self.quality_metric == 'f1':
            self.quality_metric = 'f1_1'
        if self.quality_metric == 'precision':
            self.quality_metric = 'precision_1'
        if self.quality_metric == 'recall':
            self.quality_metric = 'recall_1'

        # variables (one-charactered) and algebra are needed for simplifying expressions
        variables = list(map(chr, range(122, 122-subset_size,-1)))
        algebra = boolean.BooleanAlgebra()

        self.df_dict = {}
        for col in self.df.columns:
            self.df_dict[col] = {'data': self.df[col].values.astype(bool), 'nan_mask': pd.isna(self.df[col]).values}
        # print(type(self.y_true))
        if type(self.y_true).__module__ != 'numpy':
            self.y_true = self.y_true.values.astype(float)

        print(f'columns_subset_number: {subset_number}')

        self.all_formulas = []
        all_formulas_number = 0
        for expr, output in model_string_gen(subset_size, variables):
            simple_expr = simplify_expr(expr, subset_size, variables, algebra)
            actual_subset_size = count_actual_subset_size(simple_expr)
            number_of_binary_operators = count_operators(simple_expr)
            max_freq_of_variables = count_vars(simple_expr)

            neg_output = tuple(map(operator.not_, output))
            neg_expr = outputs_to_model_string(neg_output, subset_size, variables)
            neg_simple_expr = simplify_expr(neg_expr, subset_size, variables, algebra)
            neg_actual_subset_size = count_actual_subset_size(neg_simple_expr)
            neg_number_of_binary_operators = count_operators(neg_simple_expr)
            neg_max_freq_of_variables = count_vars(neg_simple_expr)

            summed_expr = find_sum(sums_generator(subset_size), simple_expr)
            neg_summed_expr = find_sum(sums_generator(subset_size), neg_simple_expr)
            
            readable_expr = simple_expr
            neg_readable_expr = neg_simple_expr
            for i in range(subset_size):
                readable_expr = readable_expr.replace(variables[i], f'var[{i}]')
                neg_readable_expr = neg_readable_expr.replace(variables[i], f'var[{i}]')

            # if number_of_binary_operators != neg_number_of_binary_operators or max_freq_of_variables != neg_max_freq_of_variables:
            #     print("AAAAAAAAAAAAA")
            #     print(f"{simple_expr} and {neg_simple_expr}")

            # formula_info = {}

            if self.is_formula_suitable(simple_expr, subset_size, actual_subset_size, number_of_binary_operators, max_freq_of_variables):
                self.all_formulas.append({
                    'output': np.array(output),
                    'simple_formula': simple_expr,
                    'summed_formula': summed_expr,
                    'readable_formula': readable_expr,
                    'number_of_binary_operators': number_of_binary_operators,
                    'max_freq_of_variables': max_freq_of_variables,
                    'neg_output': np.array(neg_output),
                    'neg_simple_formula': neg_simple_expr,
                    'neg_summed_formula': neg_summed_expr,
                    'neg_readable_formula': neg_readable_expr,
                    'neg_number_of_binary_operators': neg_number_of_binary_operators,
                    'neg_max_freq_of_variables': neg_max_freq_of_variables
                })
            # if self.is_formula_suitable(neg_simple_expr, subset_size, neg_actual_subset_size, neg_number_of_binary_operators, neg_max_freq_of_variables):
            #     self.all_formulas.append({
            #         'output': np.array(neg_output),
            #         'simple_formula': neg_simple_expr,
            #         'summed_formula': neg_summed_expr,
            #         'readable_formula': neg_readable_expr,
            #         'number_of_binary_operators': neg_number_of_binary_operators,
            #         'max_freq_of_variables': neg_max_freq_of_variables
            #     })
            # if formula_info:
            #     self.all_formulas.append(formula_info)

            all_formulas_number += 2
        formulas_number = len(self.all_formulas)
        print(f'formulas_number: {formulas_number}  ({formulas_number*2} with negated)')
        if self.restrict_complexity_after_size < subset_size:
            print(f'all formulas number of size {subset_size}: {all_formulas_number}')

        if self.process_number > 1:
            worker_timer = time()
            pool = Pool(self.process_number, initializer=self.worker_init)
            print(f'number of workers launched: {self.process_number}  time to launch: {time() - worker_timer}')

        start_time = time()
        self.best_models = []
        best_models_not_filtered = []
        min_quality = -1
        current_model_count = 0
        batch = []
        last_reload = False
        no_models_flag = False
        for subset_index in range(0, subset_number, batch_size):
            if subset_number - subset_index <= batch_size:
                batch.append((subset_index, subset_number, min_quality, subset_size, variables))
                current_model_count += subset_number - subset_index
                last_reload = True
            else:
                batch.append((subset_index, subset_index+batch_size, min_quality, subset_size, variables))
                current_model_count += batch_size
            
            if self.time_restriction_seconds is not None and (time() - start_time) > self.time_restriction_seconds:
                last_reload = True
            
            if len(batch) >= self.process_number or last_reload:
                # print(batch)
                # print('Workers time')
                if self.process_number == 1:
                    new_models = self.worker(*batch[0])
                else:
                    new_models = pool.starmap(self.worker, batch)
                    new_models = list(chain.from_iterable(new_models))
                self.best_models = list(chain.from_iterable([self.best_models, new_models]))
                # print('got models')
                self.best_models.sort(key=lambda row: (row[self.quality_metric], -row['number_of_binary_operators']), reverse=True)
                if self.filter_similar_between_reloads or last_reload:
                    best_models_not_filtered = deepcopy(self.best_models)
                    self.best_models = similarity_filtering(self.best_models, sim_metric, self.min_jac_score, min_same_parents)
                if crop_number is not None and not last_reload:
                    self.best_models = self.best_models[:crop_number+1]
                if len(self.best_models) == 0:
                    no_models_flag = True
                else:
                    min_quality = self.best_models[-1][self.quality_metric]
                    no_models_flag = False
                # print('sorted & filtered models')

                if not no_models_flag:
                    models_to_excel = list_to_df(self.best_models)
                    models_to_excel.drop(['columns_set', 'result'], axis=1, inplace=True)
                    beautify_simple(models_to_excel)
                    beautify_summed(models_to_excel, subset_size, variables)

                    models_to_excel = models_to_excel[['tn', 'fp', 'fn', 'tp', 'precision_1', 'recall_1', 'rocauc', 'f1_1', \
                        'nan_ratio', \
                        'accuracy', \
                        'elapsed_time', 'columns', 'summed_expr', 'readable_formula', 'number_of_binary_operators', 'max_freq_of_variables']]
                    models_to_excel.rename(columns={'precision_1': 'precision', 'recall_1': 'recall', 'f1_1': 'f1'}, inplace=True)

                    if excel_exist:
                        with pd.ExcelWriter(f"./Output/{self.project_name}_BestModels.xlsx", mode="a", if_sheet_exists='replace', engine="openpyxl") as writer:
                            models_to_excel.to_excel(writer, sheet_name=f'Size {subset_size}', index=False, freeze_panes=(1,1))
                    else:
                        models_to_excel.to_excel(f"./Output/{self.project_name}_BestModels.xlsx", index=False, freeze_panes=(1,1), sheet_name=f'Size {subset_size}')
                        excel_exist = True

                # print('wrote models to excel')
                
                batch = []
                elapsed_time = time() - start_time
                elapsed_time_per_model = elapsed_time/current_model_count
                print(f'processed_models: {current_model_count/subset_number * 100:.1f}%  elapsed_seconds: {elapsed_time:.2f}  current_quality_threshold: {min_quality:.2f}  estimated_seconds_remaining: {(subset_number - current_model_count) * elapsed_time_per_model:.2f}')
                if self.time_restriction_seconds is not None and (time() - start_time) > self.time_restriction_seconds:
                    break
        if self.process_number > 1:
            worker_timer = time()
            pool.close()
            pool.join()
            print(f'number of workers terminated: {self.process_number}  time to terminate: {time() - worker_timer}')
        if len(self.best_models) == 0:
            print(f"NO OPTIMAL MODELS FOUND ON SUBSET_SIZE={subset_size}")
        self.best_models_dict[subset_size] = deepcopy(self.best_models)
        return best_models_not_filtered


    def log_exec(self, subset_size, elapsed_time, rows_num, cols_num):
        now = datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
        if path.exists(f"./Output/{self.project_name}_log.xlsx"):
            log = pd.read_excel(f"./Output/{self.project_name}_log.xlsx")
            # search_idx = log.loc[(log['dataset'] == self.project_name) & (log['sim_metric'] == self.sim_metric) & \
            #     (log['subset_size'] == subset_size) & (log['rows_num'] == rows_num) & (log['cols_num'] == cols_num) & \
            #     (log['process_number'] == self.process_number) & (log['batch_size'] == self.batch_size) & \
            #     (log['dataset_frac'] == self.dataset_frac)].index.tolist()
            search_idx = log.loc[
                (log['dataset'] == self.project_name) & \
                (log['dataset_frac'] == self.dataset_frac) & \
                (log['subset_size'] == subset_size) & \
                (log['rows_num'] == rows_num) & \
                (log['cols_num'] == cols_num) & \
                (log['formulas_num'] == len(self.all_formulas)) & \
                (log['batch_size'] == self.batch_size) & \
                (log['process_number'] == self.process_number) & \
                (log['sim_metric'] == self.sim_metric) & \
                (log['crop_number'] == self.crop_number) & \
                (log['crop_number_in_workers'] == self.crop_number_in_workers) & \
                (log['excessive_models_num_coef'] == self.excessive_models_num_coef) & \
                (log['crop_features'] == self.crop_features)
            ].index.tolist()
            if len(search_idx) == 1:
                log.loc[search_idx, ['elapsed_time']] = elapsed_time
                log.loc[search_idx, ['datetime']] = now
                with pd.ExcelWriter('./Output/log.xlsx', mode="w", engine="openpyxl") as writer:
                    log.to_excel(writer, sheet_name='Logs', index=False, freeze_panes=(1,1))
            else:
                new_row = pd.DataFrame(data={
                                            'datetime': [now],
                                            'dataset': [self.project_name],
                                            'dataset_frac': [self.dataset_frac],
                                            'subset_size': [subset_size],
                                            'rows_num': [rows_num],
                                            'cols_num': [cols_num],
                                            'formulas_num': [len(self.all_formulas)],
                                            'batch_size': [self.batch_size],
                                            'process_number': [self.process_number],    
                                            'elapsed_time': [elapsed_time],
                                            'sim_metric': [self.sim_metric],
                                            'crop_number': [self.crop_number],
                                            'crop_number_in_workers': [self.crop_number_in_workers],
                                            'excessive_models_num_coef': [self.excessive_models_num_coef],
                                            'crop_features': [self.crop_features],
                                            'complexity_restr_operators': [self.complexity_restr_operators],
                                            'complexity_restr_vars': [self.complexity_restr_vars],
                                            'time_restriction_seconds': [self.time_restriction_seconds]
                                            })
                log = pd.concat([log, new_row], ignore_index=True)
                with pd.ExcelWriter(f"./Output/{self.project_name}_log.xlsx", mode="w", engine="openpyxl") as writer:
                    log.to_excel(writer, sheet_name='Logs', index=False, freeze_panes=(1,1))
        else:
            log = pd.DataFrame(data={
                                     'datetime': [now],
                                     'dataset': [self.project_name],
                                     'dataset_frac': [self.dataset_frac],
                                     'subset_size': [subset_size],
                                     'rows_num': [rows_num],
                                     'cols_num': [cols_num],
                                     'formulas_num': [len(self.all_formulas)],
                                     'batch_size': [self.batch_size],
                                     'process_number': [self.process_number],    
                                     'elapsed_time': [elapsed_time],
                                     'sim_metric': [self.sim_metric],
                                     'crop_number': [self.crop_number],
                                     'crop_number_in_workers': [self.crop_number_in_workers],
                                     'excessive_models_num_coef': [self.excessive_models_num_coef],
                                     'crop_features': [self.crop_features],
                                     'complexity_restr_operators': [self.complexity_restr_operators],
                                     'complexity_restr_vars': [self.complexity_restr_vars],
                                     'time_restriction_seconds': [self.time_restriction_seconds]
                                     })
            with pd.ExcelWriter(f"./Output/{self.project_name}_log.xlsx", mode="w", engine="openpyxl") as writer:
                    log.to_excel(writer, sheet_name='Logs', index=False, freeze_panes=(1,1))