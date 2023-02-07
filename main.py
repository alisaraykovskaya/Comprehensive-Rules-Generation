from loadData import LoadData
from binarizerClass import Binarizer
from CRGmodel import CRG
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

config = {
    "load_data_params":{
        "project_name": "DivideBy30Remainder", 
        "load_from_pkl": False
    },

    "rules_generation_params": {
        "quality_metric": "f1", #'f1', 'accuracy', 'rocauc', 'recall', 'precision'
        "subset_size": 3,

        "process_number": 10, # int or "default" = 90% of cpu
        "batch_size": 10000, # number of subsets, which each worker will be processing on every reload
        "filter_similar_between_reloads": False, # If true filter similar models between reloads, otherwise saved in excel best_models between reloads will contain similar models. May lead to not reproducible results.
        
        "crop_number": 1000, # number of best models to compute quality metric threshold
        "crop_number_in_workers": 1000, # same like crop_number, but within the workres. If less than crop_number, it may lead to unstable results
        "excessive_models_num_coef": 3, # how to increase the actual crop_number in order to get an appropriate number of best models after similarity filtering (increase if the result contains too few models)
        
        "dataset_frac": 1,
        "crop_features": 100, # the number of the most important features to remain in a dataset. Needed for reducing working time if dataset has too many features

        "incremental_run": True,
        "crop_features_after_size": 1,
    },
  
    "similarity_filtering_params": {
        "sim_metric": "PARENT", #"JAC_SCORE"
        "min_jac_score": 0.9, #JAC_SCORE threshold
        "min_same_parents": 2 #PARENT sim threshold
    },
  
    "binarizer_params": {
        "unique_threshold": 20, # maximal number of unique values to consider numerical variable as category
        "q": 20, # number of quantiles to split numerical variable, can be lowered if there is need in speed
        "exceptions_threshold": 0.01, # max % of exeptions allowed while converting a variable into numeric to treat it as numeric 
        "numerical_binarization": "range", #"range"
        "nan_threshold": 0.9, # max % of missing values allowed to process the variable
        "share_to_drop": 0.005, # max % of zeros allowed for a binarized column or joint % of ones for the the most unballanced columns which are joined together into 'other' category.
        "create_nan_features": True # If true for every feature that contains nans, corresponding nan indecatore feature will be created
    } 
}

def main():
    print('LOADING DATA...')
    df = LoadData(**config["load_data_params"])
    y_true = df['Target']
    df.drop('Target', axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df, y_true, test_size=0.2, stratify=y_true, random_state=12)

    binarizer = Binarizer(**config["load_data_params"], **config["binarizer_params"])
    crg_alg = CRG(binarizer, **config["load_data_params"], **config["rules_generation_params"], **config["similarity_filtering_params"])

    crg_alg.fit(X_train, y_train)
    preds = crg_alg.predict(X_test, 5)
    print(classification_report(y_test, preds))


if __name__=='__main__':
    main()