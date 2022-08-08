It's better to prepare new environment for execution. It can be done with:
1) conda via yml file: ```conda env create -f boolean_ml_env.yml```
2) pip: ```pip install -r boolean_ml_env.txt```

---

Parameters are located in ```main.py```:

### Models settings

- ```subset_size``` - size of columns subset which will be considered in formulas, i.e. number of variables in boolean formulas
- ```parallel``` - flag of parallel execution
- ```process_number``` - number of processes for paralleling
- ```batch_size``` - size of batch of models to be processed by workers

### Data preprocessing settings
- ```pkl_reload``` - if True then dataset will be loaded, binarized and saved as pickle, otherwise it will be loaded from pickle
- ```boolarize``` - boolarize means turning all 1 to True, 0 to False in dataframe

### Binarizer settings
```unique_threshold``` - maximal number of unique values to consider numerical variable as category
```q``` - number of quantiles to split numerical variable
```exceptions_threshold``` - max % of exeptions allowed while converting a variable into numeric to treat it as numeric 
```numerical_binarization``` - type of numerical binarization (range or threshold)
```nan_threshold``` - max % of missing values allowed to process the variable
```share_to_drop``` - max % of zeros allowed for a binarized column or joint % of ones for the the most unballanced columns (with the least number of ones) which are joined together into 'other' category.

---

To run:

```
python main.py
```
