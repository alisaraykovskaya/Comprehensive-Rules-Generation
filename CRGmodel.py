from os import path, mkdir
import platform
from time import time
from math import factorial
from multiprocessing import freeze_support, cpu_count, Pool
from itertools import combinations, chain, islice
from copy import deepcopy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import boolean
from numba import njit

from utils import model_string_gen, simplify_expr, find_sum, sums_generator, similarity_filtering, list_to_df, post_simplify, beautify_simple, beautify_summed
from utils import get_1var_importance_order, get_feature_importance, add_missing_features
from metrics_utils import count_actual_subset_size, count_operators, count_vars, count_confusion_matrix, calculate_metrics_for, get_parent_set
from metrics_utils import negate_model
from best_models import make_nan_mask, apply_nan_mask_list


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


    def fit(self, df, y_true):
        self.raw_df = df.copy()
        self.raw_df['Target'] = y_true

        print('BINARIZING DATA...')
        self.df = self.binarizer.fit_transform(self.raw_df)

        if self.dataset_frac != 1:
            self.df, _ = train_test_split(df, train_size=self.dataset_frac, stratify=self.df['Target'], random_state=12)

        self.y_true = self.df['Target']
        self.df = self.df.drop(columns=['Target'])
        self.columns_ordered = self.df.columns
        columns_number = len(self.columns_ordered)

        print(f'project_name: {self.project_name}  columns_number: {columns_number}  observations_number: {self.df.shape[0]}')

        print('\nRUNNING ON subset_size=1 TO DETERMINE INITIAL FEATURE IMPORTANCES')
        start_time = time()
        best_1_variable = self.find_best_models(is_onevar=True)
        elapsed_time = time() - start_time
        self.log_exec(subset_size=1, elapsed_time=elapsed_time, rows_num=self.df.shape[0], cols_num=self.df.shape[1])
        self.columns_ordered = get_1var_importance_order(best_1_variable, subset_size=1, columns_number=columns_number)
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

    
    def predict(self, raw_df_test, k_best):
        print('BINARIZING DATA...')
        df_test = self.binarizer.transform(raw_df_test)
        df_test_dict = {}
        for col in df_test.columns:
            df_test_dict[col] = {'data': df_test[col].values.astype(bool), 'nan_mask': pd.isna(df_test[col]).values}
        result = []
        
        if isinstance(k_best, int):
            models = self.best_models[:k_best]
            for model_dict in models:
                columns = model_dict['columns']
                nan_mask = np.full_like(df_test.shape[0], True, dtype=bool)
                df_np_cols = []
                df_nan_cols = []
                for col in columns:
                    df_np_cols.append(df_test_dict[col]['data'])
                    df_nan_cols.append(df_test_dict[col]['nan_mask'])
                df_np_cols = np.array(df_np_cols)
                df_nan_cols = np.array(df_nan_cols)

                expr = model_dict['expr']
                model_template = eval(f'njit(lambda df_np_cols: {expr})')
                tmp = model_template(df_np_cols)
                result.append(tmp)
            result = np.stack(result, axis=-1)
            print(result.shape)
            result = np.rint(np.mean(result, axis=1)).astype(bool)
        return result
            



        # if k_best is int:
        #     take model with k_best id and predict
        # if k_best is list:
        #     make an ensemble of models and predict

        # self.y_true


    def worker_init(self):
        self.start_time = time()


    def worker(self, start_idx, end_idx, min_quality, subset_size):
        best_models = []
        for columns in islice(combinations(self.columns_ordered, subset_size), start_idx, end_idx):
            columns = list(columns)
            nan_mask = np.full_like(self.y_true, True, dtype=bool)
            df_np_cols = []
            df_nan_cols = []
            for col in columns:
                df_np_cols.append(self.df_dict[col]['data'])
                df_nan_cols.append(self.df_dict[col]['nan_mask'])
            df_np_cols = np.array(df_np_cols)
            df_nan_cols = np.array(df_nan_cols)

            for formula_dict in self.all_formulas:
                formula_template = formula_dict['formula_template']
                expr = formula_dict['expr']
                summed_expr = formula_dict['summed_expr']
                result = formula_template(df_np_cols)
                nan_mask = make_nan_mask(nan_mask, df_nan_cols, subset_size)
                nan_count = np.count_nonzero(nan_mask)
                result = apply_nan_mask_list(result, nan_mask)

                tp, fp, fn, tn = count_confusion_matrix(self.y_true, result)
                precision, recall, f1, rocauc, accuracy = calculate_metrics_for(tp, fp, fn, tn)

                if self.quality_metric == 'f1_1':
                    quality = f1
                elif self.quality_metric == 'precision_1':
                    quality = precision
                elif self.quality_metric == 'recall_1':
                    quality = recall
                elif self.quality_metric == 'rocauc':
                    quality = rocauc
                elif self.quality_metric == 'accuracy':
                    quality = accuracy

                if quality > min_quality:
                    precision_0 = 0 if (tn + fn) == 0 else tn / (tn + fn)
                    recall_0 = 0 if (tn + fp) == 0 else tn / (tn + fp)
                    f1_0 = 0 if (precision_0 + recall_0) == 0 else 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
                    columns_set = get_parent_set(columns)
                    elapsed_time = time() - self.start_time
                    simple_formula = expr
                    for i in range(subset_size):
                        simple_formula = simple_formula.replace(f'df_np_cols[{i}]', columns[i])
                    model_info = {'f1_1': f1, 'f1_0': f1_0, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'precision_1': precision, 'precision_0': precision_0,\
                        'recall_1': recall, 'recall_0': recall_0, 'rocauc': rocauc, 'accuracy': accuracy, 'elapsed_time': elapsed_time, 'nan_ratio': nan_count/self.y_true.shape[0], \
                        'summed_expr': summed_expr, 'columns': columns, 'expr': expr, 'expr_len': len(expr), 'is_negated': 'False',  'simple_formula': simple_formula, 'columns_set': columns_set, \
                        'number_of_binary_operators': formula_dict['number_of_binary_operators'], 'max_freq_of_variables': formula_dict['max_freq_of_variables']}
                    if self.sim_metric == 'JAC_SCORE':
                        model_info['result'] = result
                    else:
                        model_info['result'] = None
                    best_models.append(model_info)
                result_neg = negate_model(result)
                tp, fp, fn, tn = count_confusion_matrix(self.y_true, result_neg)
                precision, recall, f1, rocauc, accuracy = calculate_metrics_for(tp, fp, fn, tn)
                if self.quality_metric == 'f1_1':
                    quality = f1
                elif self.quality_metric == 'precision_1':
                    quality = precision
                elif self.quality_metric == 'recall_1':
                    quality = recall
                elif self.quality_metric == 'rocauc':
                    quality = rocauc
                elif self.quality_metric == 'accuracy':
                    quality = accuracy
                if quality > min_quality:
                    precision_0 = 0 if (tn + fn) == 0 else tn / (tn + fn)
                    recall_0 = 0 if (tn + fp) == 0 else tn / (tn + fp)
                    f1_0 = 0 if (precision_0 + recall_0) == 0 else 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
                    columns_set = get_parent_set(columns)
                    elapsed_time = time() - self.start_time
                    simple_formula = expr
                    for i in range(subset_size):
                        simple_formula = simple_formula.replace(f'df_np_cols[{i}]', columns[i])
                    expr = '~(' + expr + ')'
                    simple_formula = '~(' + simple_formula + ')'
                    model_info = {'f1_1': f1, 'f1_0': f1_0, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'precision_1': precision, 'precision_0': precision_0,\
                        'recall_1': recall, 'recall_0': recall_0, 'rocauc': rocauc, 'accuracy': accuracy, 'elapsed_time': elapsed_time, 'nan_ratio': nan_count/self.y_true.shape[0], \
                        'summed_expr': summed_expr, 'columns': columns, 'expr': expr, 'expr_len': len(expr), 'is_negated': 'True', 'simple_formula': simple_formula, 'columns_set': columns_set, \
                        'number_of_binary_operators': formula_dict['number_of_binary_operators'], 'max_freq_of_variables': formula_dict['max_freq_of_variables']}
                    if self.sim_metric == 'JAC_SCORE':
                        model_info['result'] = result
                    else:
                        model_info['result'] = None
                    best_models.append(model_info)


                if self.crop_number_in_workers is not None and len(best_models) >= self.crop_number_in_workers * self.excessive_models_num_coef:
                    best_models.sort(key=lambda row: (row[self.quality_metric], -row['number_of_binary_operators']), reverse=True)
                    best_models = best_models[:self.crop_number_in_workers]
                    min_quality = best_models[-1][self.quality_metric]

        return best_models


    def find_best_models(self, is_onevar):
        excel_exist = False
        if path.exists(f"./Output/BestModels_{self.project_name}_train.xlsx"):
            excel_exist = True

        if is_onevar:
            subset_size = 1
            crop_number = self.onevar_crop_number
            sim_metric = self.onevar_sim_metric
            min_same_parents = self.onevar_min_same_parents
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

        self.all_formulas = []
        for expr in model_string_gen(subset_size):        
            simple_expr = simplify_expr(expr, subset_size, variables, algebra)
            # If formula is a tautology
            if simple_expr == '1':
                continue
            actual_subset_size = count_actual_subset_size(simple_expr)
            if actual_subset_size < subset_size:
                continue
            number_of_binary_operators = count_operators(simple_expr)
            max_freq_of_variables = count_vars(simple_expr)

            summed_expr = find_sum(sums_generator(subset_size), simple_expr)

            # Replace one-charachter variables in simplified expr with 'df[columns[{i}]]'
            # for easy execution
            for i in range(subset_size):
                simple_expr = simple_expr.replace(variables[i], f'df_np_cols[{i}]')

            self.all_formulas.append({
                'formula_template': eval(f'njit(lambda df_np_cols: {simple_expr})'),
                'expr': simple_expr,
                'summed_expr': summed_expr,
                'number_of_binary_operators': number_of_binary_operators,
                'max_freq_of_variables': max_freq_of_variables
            })
        formulas_number = len(self.all_formulas)
        total_model_count = formulas_number * subset_number
        print(f'formulas_number: {formulas_number}  columns_subset_number(models_per_formula): {subset_number}  total_count_of_models: {total_model_count}')

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
        for subset_index in range(0, subset_number, batch_size):
            if subset_number - subset_index <= batch_size:
                batch.append((subset_index, subset_number, min_quality, subset_size))
                current_model_count += (subset_number - subset_index) * formulas_number
                last_reload = True
            else:
                batch.append((subset_index, subset_index+batch_size, min_quality, subset_size))
                current_model_count += batch_size * formulas_number
            
            if len(batch) >= self.process_number or last_reload:
                # print(batch)
                # print('Workers time')
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
                min_quality = self.best_models[-1][self.quality_metric]
                # print('sorted & filtered models')

                models_to_excel = list_to_df(self.best_models)
                models_to_excel.drop(['columns_set', 'result'], axis=1, inplace=True)
                # models_to_excel = post_simplify(models_to_excel, subset_size, variables, algebra)
                models_to_excel['simple_formula'] = models_to_excel.apply(lambda x: post_simplify(x, subset_size, variables, algebra), axis=1)
                beautify_simple(models_to_excel)
                beautify_summed(models_to_excel, subset_size, variables)

                models_to_excel = models_to_excel[['tn', 'fp', 'fn', 'tp', 'precision_1', 'recall_1', 'rocauc', 'f1_1', 'nan_ratio', 'accuracy', \
                    'elapsed_time', 'columns', 'summed_expr', 'is_negated', 'simple_formula', 'number_of_binary_operators', 'max_freq_of_variables']]
                models_to_excel.rename(columns={'precision_1': 'precision', 'recall_1': 'recall', 'f1_1': 'f1'}, inplace=True)

                if excel_exist:
                    with pd.ExcelWriter(f"./Output/BestModels_{self.project_name}_train.xlsx", mode="a", if_sheet_exists='replace', engine="openpyxl") as writer:
                        models_to_excel.to_excel(writer, sheet_name=f'Size {subset_size}', index=False, freeze_panes=(1,1))
                else:
                    models_to_excel.to_excel(f"./Output/BestModels_{self.project_name}_train.xlsx", index=False, freeze_panes=(1,1), sheet_name=f'Size {subset_size}')
                    excel_exist = True

                # print('wrote models to excel')
                
                batch = []
                elapsed_time = time() - start_time
                elapsed_time_per_model = elapsed_time/current_model_count
                print(f'processed_models: {current_model_count/total_model_count * 100:.1f}%  elapsed_seconds: {elapsed_time:.2f}  current_quality_threshold: {min_quality:.2f}  estimated_seconds_remaining: {(total_model_count - current_model_count) * elapsed_time_per_model:.2f}')
        worker_timer = time()
        pool.close()
        pool.join()
        print(f'number of workers terminated: {self.process_number}  time to terminate: {time() - worker_timer}')
        return best_models_not_filtered


    def log_exec(self, subset_size, elapsed_time, rows_num, cols_num):
        if path.exists("./Output/log.xlsx"):
            log = pd.read_excel('./Output/log.xlsx')
            search_idx = log.loc[(log['dataset'] == self.project_name) & (log['sim_metric'] == self.sim_metric) & \
                (log['subset_size'] == subset_size) & (log['rows_num'] == rows_num) & (log['cols_num'] == cols_num) & \
                (log['process_number'] == self.process_number) & (log['batch_size'] == self.batch_size) & \
                (log['dataset_frac'] == self.dataset_frac)].index.tolist()
            if len(search_idx) == 1:
                log.loc[search_idx, ['elapsed_time']] = elapsed_time
                with pd.ExcelWriter('./Output/log.xlsx', mode="w", engine="openpyxl") as writer:
                    log.to_excel(writer, sheet_name='Logs', index=False, freeze_panes=(1,1))
            else:
                new_row = pd.DataFrame(data={'dataset': [self.project_name], 'dataset_frac': [self.dataset_frac], 'rows_num': [rows_num], 'cols_num': [cols_num], 'sim_metric': [self.sim_metric], \
                    'subset_size': [subset_size], 'batch_size': [self.batch_size], 'process_number': [self.process_number], 'elapsed_time': [elapsed_time]})
                log = pd.concat([log, new_row], ignore_index=True)
                with pd.ExcelWriter('./Output/log.xlsx', mode="w", engine="openpyxl") as writer:
                    log.to_excel(writer, sheet_name='Logs', index=False, freeze_panes=(1,1))
        else:
            log = pd.DataFrame(data={'dataset': [self.project_name], 'dataset_frac': [self.dataset_frac], 'rows_num': [rows_num], 'cols_num': [cols_num], 'sim_metric': [self.sim_metric], \
                'subset_size': [subset_size], 'batch_size': [self.batch_size], 'process_number': [self.process_number], 'elapsed_time': [elapsed_time]})
            with pd.ExcelWriter('./Output/log.xlsx', mode="w", engine="openpyxl") as writer:
                    log.to_excel(writer, sheet_name='Logs', index=False, freeze_panes=(1,1))