It's better to prepare new environment for execution. It can be done with:
1) conda via yml file: ```conda env create -f CRG_env.yml```
2) pip (Python 3.9): ```pip install -r CRG_env.txt```

---

### Main Pipeline
1. Data loading and binarizing
2. Determining feature importance by running main algorithm on ```subset_size=1```
3. Reordering (possibly croping) features by importance, in order to increase the chance of getting best models faster
4. Running main alggorithm

### Main Algorithm
There are 2 types of processes: main (only one) and workers (many). Main process generates boolean formulas, feed them to workers and gather results from them. Workers take formula and by generating all permutations of length ```subset_size``` of df features evaluate all possible models on given formula. This can be (should be) done in batches. Batch is defined by ```process_number``` and ```formula_per_worker```, for example if we have 2 workers and we want each of them to process 2 formulas then batch will consist of 4 formulas. Iteration of creating batch of formulas, feeding it to workers and getting results we usually call a reload. Reloads help a lot with showing progress of algorithm and creating intermediate results between reloads.

Depending on size of dataset and ```subset_size``` number of models can grow exponentially, this affects not only running time, but memory consumption too. To address this issues there are few solutions:
1. ```min_quality``` - threshold for entry in best_models table, which is updated between reloads and locally in workers
2. Reloads update ```min_quality``` for all workers globally, but creating and preparing pool of workers can take some time, so it is better to keep moderate frequency of reloads
3. Cropping accumulated number of models after reloads and in workers if number of models exceed certain amount. Implemented using flags ```crop_number```, ```crop_number_in_workers```, ```excessive_models_num_coef```
4. Dropping nan only before models for speceific subset of df is correct practise, but can be very slow. By using ```dropna_on_whole_df``` all nan can be droped in the beginning
5. If dataset is too large ```crop_features``` and binarizer's ```q``` parameters can be used to reduce number of features.

### Rules generation settings 
- ```quality_metric``` - options: ('f1', 'accuracy', 'rocauc', 'recall', 'precision'). Quality metric by which the models will be sorted and the entry threshold into best_models table will be selected
- ```subset_size``` - size of columns subset which will be considered in formulas, i.e. number of variables in boolean formulas
- ```process_number``` - number of processes for parallel execution (int or "defalut" = 90% of cpu)
- ```formula_per_worker``` - number of formulas passed to each worker in each batch
- ```crop_features``` - leave only first ```crop_features``` important features in dataset. Needed for reducing working time if dataset has too many features
- ```crop_number``` - leave only best ```crop_number``` models in best_models table after getting all outputs from workers between reloads
- ```crop_number_in_workers``` - leave only best ```crop_number_in_workers``` models in worker's local best_models table, when worker accumulated ```crop_number_in_workers * excessive_models_num_coef``` models. Almost in every case should be equal to ```crop_number```, but when there is need in memory and speed can be lowered.
- ```excessive_models_num_coef``` - coefficient that indicates how many models should be accumulated by worker before cropping. Exact number is calculated following way: ```crop_number_in_workers * excessive_models_num_coef```
- ```dropna_on_whole_df``` - If ```True``` then dropna will be performed on whole dataset before algorithm, otherwise dropna will be used for every model individually to it's subset of columns (which is slow). Almost in every case should be ```False```, but when there is need in speed can be set ```True```.
- ```desired_minutes_per_worker``` - how many minutes would you like each worker to run. Using this parameter needed ```formula_per_worker``` for workers to run **approximately** ```desired_minutes_per_worker``` will be printed


### Data loading settings
- ```pkl_reload``` - if True then dataset will be loaded, binarized and saved as pickle, otherwise it will be loaded from pickle
- ```file_name``` - name of original dataset file
- ```target_name``` - name of target column in original dataset
- ```file_ext``` - original dataset file extension 


### Similar models filtering settings 
-   ```sim_metric``` - options: ("PARENT" or "JAC_SCORE"). Parent similarity decides if models are similar if they have ```min_same_parents``` common features, Jaccard similarity  decides if models are similar if their Jaccard similarity coef is higher than ```min_jac_score```
-   ```min_jac_score``` - threshold for "JAC_SCORE" similarity, ```default=0.9```
-   ```min_same_parents``` - threshold for "PARENT" similarity


### Binarizer settings
- ```unique_threshold``` - maximal number of unique values to consider numerical variable as category
- ```q``` - number of quantiles to split numerical variable, can be lowered if there is need in speed
- ```exceptions_threshold``` - max % of exeptions allowed while converting a variable into numeric to treat it as numeric 
- ```numerical_binarization``` - type of numerical binarization ("range" or "threshold")
- ```nan_threshold``` - max % of missing values allowed to process the variable
- ```share_to_drop``` - max % of zeros allowed for a binarized column or joint % of ones for the the most unballanced columns (with the least number of ones) which are joined together into 'other' category.

---

To run:

```
python main.py
```
