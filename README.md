It's better to prepare new environment for execution. It can be done with:
1) conda via yml file: ```conda env create -f CRG_env.yml```
2) pip (Python 3.9): ```pip install -r CRG_env.txt```

---

Parameters are located in ```config.json```:

### Rules generation settings 
- ```quality_metric``` - ('f1', 'accuracy', 'rocauc', 'recall', 'precision')
- ```subset_size``` - size of columns subset which will be considered in formulas, i.e. number of variables in boolean formulas
- ```process_number``` - number of processes for parallel execution (int or "defalut" = 90% of cpu)
- ```formula_per_worker``` - number of boolean formulas passed to workers 
- ```crop_number``` - 
- ```crop_number_in_workers``` - 
- ```filter_similar_between_reloads``` - 
- ```workers_filter_similar``` - 


### Data loading settings
- ```pkl_reload``` - if True then dataset will be loaded, binarized and saved as pickle, otherwise it will be loaded from pickle
- ```file_name``` - name of original dataset file
- ```target_name``` - name of target column in original dataset
- ```file_ext``` - original dataset file extension 


### Similar models filtering settings 
-   ```sim_metric``` - ("PARENT" or "JAC_SCORE"),
-   ```min_jac_score``` - 0.9,
-   ```min_same_parents``` - 2


### Binarizer settings
- ```unique_threshold``` - maximal number of unique values to consider numerical variable as category
- ```q``` - number of quantiles to split numerical variable
- ```exceptions_threshold``` - max % of exeptions allowed while converting a variable into numeric to treat it as numeric 
- ```numerical_binarization``` - type of numerical binarization ("range" or "threshold")
- ```nan_threshold``` - max % of missing values allowed to process the variable
- ```share_to_drop``` - max % of zeros allowed for a binarized column or joint % of ones for the the most unballanced columns (with the least number of ones) which are joined together into 'other' category.

---

To run:

```
python main.py
```
