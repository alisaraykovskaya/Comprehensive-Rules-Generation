from loadData import LoadData
from binarizerClass import Binarizer
from CRGmodel import CRG
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from copy import deepcopy


dataset_names = [
    "heart",
    # "DivideBy30",
    # "DivideBy30Remainder"
    "diabetes",
    "churn",
    "bank",
    "hepatitis",
]

config = {
    "load_data_params":{
        "project_name": "heart", 
        "load_from_pkl": False
    },

    "rules_generation_params": {
        "quality_metric": "f1", # 'f1', 'accuracy', 'rocauc', 'recall', 'precision'
        "subset_size": 4, # number of variables for boolean formula in models

        "process_number": "default", # int or "default" = 90% of cpu
        "batch_size": 10000, # number of subsets, which each worker will be processing on every reload. number_of_models = batch_size * formula_number
        "filter_similar_between_reloads": True, # If true filter similar models between reloads, otherwise saved in excel between reloads best_models will contain similar models. May lead to not reproducible results.
        
        "crop_number": 4000, # crop number of best models after every reload
        "crop_number_in_workers": 4000, # crop the number of best models in worker if it accumulated more than (crop_number_in_workers * excessive_models_num_coef) models. If less than crop_number, it may lead to unstable results
        "excessive_models_num_coef": 3, # used for computing cropping threshold in workers
        
        "dataset_frac": 1, # use only fraction of training dataset, use this if algorithm running too long
        "crop_features": 80, # the number of the most important features to remain in a dataset. Needed for reducing working time if dataset has too many features
        "crop_parent_features": 80,

        "complexity_restr_operators": "subset_size+1", # Consider Boolean formulas, only with number of binary operators less or equal than a given number. It is worth noting that the value should not be less than subset_size-1
        "complexity_restr_vars": None, # Consider Boolean formulas only with number of repetitions of one variable less or equal than a given number

        "time_restriction_seconds": 500000, # Limiting the running time of the algorithm per subset_size

        "incremental_run": True, # run algorithm on subset_size=1...subset_size
        "crop_features_after_size": 2, # if crop_features and incremental_run, then cropping will occur after subset_size
        "restrict_complexity_after_size": 2,
    },
  
    "similarity_filtering_params": {
        "sim_metric": "PARENT", # "JAC_SCORE"
        "min_jac_score": 0.9, # JAC_SCORE threshold
        "min_same_parents": "subset_size" # PARENT sim threshold
    },
  
    "binarizer_params": {
        "unique_threshold": 20, # maximal number of unique values to consider numerical variable as category
        "q": 20, # number of quantiles to split numerical variable, can be lowered if there is need in speed
        "exceptions_threshold": 0.05, # max % of exeptions allowed while converting a variable into numeric to treat it as numeric 
        "numerical_binarization": "range", # "range" "threshold"
        "nan_threshold": 0.9, # max % of missing values allowed to process the variable
        "share_to_drop": 0.01, # max % of zeros allowed for a binarized column or joint % of ones for the the most unballanced columns which are joined together into 'other' category.
        "create_nan_features": True # If true for every feature that contains nans, corresponding nan indecatore feature will be created
    } 
}

already_splitted_datasets = set(["heart", "bank", "churn", "diabetes", "hepatitis"])


def main():
    for dataset_name in dataset_names:
        config["load_data_params"]["project_name"] = dataset_name
        print('LOADING DATA...')
        if config["load_data_params"]["project_name"] in already_splitted_datasets:
            df_train_config = deepcopy(config["load_data_params"])
            df_train_config["project_name"] += "_train"
            df = LoadData(**df_train_config)
            y_train = df['Target']
            X_train = df.drop('Target', axis=1)

            df_test_config = deepcopy(config["load_data_params"])
            df_test_config["project_name"] += "_test"
            X_test = LoadData(**df_test_config)
            y_test = X_test['Target']
            X_test.drop('Target', axis=1, inplace=True)
        else:
            df = LoadData(**config["load_data_params"])
            y_true = df['Target']
            df.drop('Target', axis=1, inplace=True)
            X_train, X_test, y_train, y_test = train_test_split(df, y_true, test_size=0.2, stratify=y_true, random_state=12)

        binarizer = Binarizer(**config["load_data_params"], **config["binarizer_params"])
        crg_alg = CRG(binarizer, **config["load_data_params"], **config["rules_generation_params"], **config["similarity_filtering_params"])

        crg_alg.fit(X_train, y_train)
        # for subset_size in range(1, config['rules_generation_params']['subset_size']+1):
        #     print(f'subset_size={subset_size}')
        #     preds = crg_alg.predict(raw_df_test=X_test, subset_size=subset_size, k_best=5)
        #     print(classification_report(y_test, preds))

        crg_alg.predict(raw_df_test=X_test, y_test=y_test, subset_size=config['rules_generation_params']['subset_size'], incremental=True)



if __name__=='__main__':
    main()