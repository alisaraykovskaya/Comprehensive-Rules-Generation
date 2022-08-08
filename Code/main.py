from loadData import LoadData
from binarizer import binarize_df
from best_models import find_best_model, add_metrics, tupleList_to_df, validate_model, get_formulas, beautify_formulas

from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support, cpu_count
from line_profiler_pycharm import profile
import pandas as pd
import fastmetrics

import os.path


# Returns True if two models are so similar one should be filtered, False otherwise
# Parameter metric is a string describing the similarity metric to be used
def compare_model_similarity(model1, model2, metric, min_jac_score=.9):
    if metric == "JAC_SCORE":
        return fastmetrics.fast_jaccard_score(model1, model2) >= min_jac_score


@profile
def main():
    """ Dataset file name settings """
    # file_name = 'COVID_X_train'
    # target_name = 'y'
    # file_ext = 'xlsx'

    # file_name = 'noisy_COVID'
    # target_name = 'y'
    # file_ext = 'csv'

    # file_name = 'heart'
    # target_name = 'HeartDisease'
    # file_ext = 'csv'

    # file_name = 'heart_2020'
    # target_name = 'HeartDisease'
    # file_ext = 'csv'

    # file_name = 'breast-cancer'
    # target_name = 'diagnosis'
    # file_ext = 'csv'

    # file_name = 'water_potability'
    # target_name = 'Potability'
    # file_ext = 'csv'

    # file_name = 'cardio_train'
    # target_name = 'cardio'
    # file_ext = 'csv'

    # file_name = 'hospital_death'
    # target_name = 'hospital_death'
    # file_ext = 'csv'

    # file_name = 'titanic'
    # target_name = 'Survived'
    # file_ext = 'csv'

    file_name = 'DivideBy30'
    target_name = 'Divisible by 30'
    file_ext = 'csv'

    # file_name = 'Data_Miocarda'
    # target_name = 'Outcome_113_Atrial_fibrillation_'
    # file_ext = 'xlsx'

    """ Changeable model settings """
    subset_size = 2
    pkl_reload = False
    min_jac_score = .90

    """ Binarizer settings"""
    unique_threshold = 20
    q = 15
    exceptions_threshold = 0.01
    numerical_binarization = 'range'
    nan_threshold = 0.8
    share_to_drop = 0.05

    """Parallel Settings"""
    num_threads = cpu_count()
    parallel = True

    print('Loading data...')
    if not pkl_reload and not os.path.exists(f'./Data/{file_name}_binarized.pkl'):
        print('Binarized data was not found')
        pkl_reload = True
    if pkl_reload:
        df = LoadData(file_name, file_ext, target_name, pkl_reload=pkl_reload)
        print('Binarizing data...')
        df = binarize_df(df, unique_threshold=unique_threshold, q=q, exceptions_threshold=exceptions_threshold,
                         numerical_binarization=numerical_binarization, nan_threshold=nan_threshold,
                         share_to_drop=share_to_drop)
        df.to_pickle(f'./Data/{file_name}_binarized.pkl')
    else:
        print('Data was loaded from pickle')
        df = pd.read_pickle(f'./Data/{file_name}_binarized.pkl')

    y_true = df['Target']
    df.drop('Target', axis=1, inplace=True)
    stratify = y_true
    X_train, X_test, y_train, y_test = train_test_split(df, y_true, test_size=0.3, stratify=stratify, random_state=12)

    print('Begin training...')
    best_formulas = find_best_model(X_train, y_train, subset_size, parallel, num_threads)
    best_formulas = sorted(best_formulas, key=lambda tup: tup[0], reverse=True)

    i = 0

    while i < len(best_formulas):
        j = i+1

        while j < len(best_formulas):
            if compare_model_similarity(best_formulas[i][3], best_formulas[j][3], min_jac_score):
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
    models.to_excel(f"BestModels_{file_name}.xlsx", index=False, freeze_panes=(1, 1))


if __name__=='__main__':
    # Windows flag
    freeze_support()
    main()
