import pandas as pd
import os.path as path
import re
import numpy as np
import random

def change_column_name(string):
  #remove leading and trailing characters (spaces)
  string = re.sub('^ | $','', string)
  #substitute spaces with _ and leave only letters, digits and _ 
  string = re.sub('\W+','', string.replace(' ','_'))
  return string

def LoadData(project_name, load_from_pkl=False):
    
    if project_name == "DivideBy30":
      df_name = "DivideBy30"
      target_name = "Div By 30"
      file_ext = "xlsx" 
    
    elif project_name == "DivideBy30RemainderNoiseNull":
      df_name = "DivideBy30RemainderNoiseNull"
      target_name = "Div By 30"
      file_ext = "xlsx"

    elif project_name == "DivideBy30RemainderNull":
      df_name = "DivideBy30RemainderNull"
      target_name = "Div By 30"
      file_ext = "xlsx"

    elif project_name == "DivideBy30RemainderNoise":
      df_name = "DivideBy30RemainderNoise"
      target_name = "Div By 30"
      file_ext = "xlsx"

    elif project_name == "DivideBy30Remainder":
      df_name = "DivideBy30Remainder"
      target_name = "Div By 30"
      file_ext = "xlsx"

    elif project_name == "Titanic":
      df_name = "titanic"
      target_name = "Survived"
      file_ext = "csv" 

    elif project_name == "TitanicNullSex":
      df_name = "titanic_null_sex"
      target_name = "Survived"
      file_ext = "csv" 

    elif project_name == "heart":
      df_name = "heart"
      target_name = "HeartDisease"
      file_ext = "csv" 

    elif project_name == "ApptsAnon":
      df_name = "Missed_Pedi_Appts_v4.Anon"
      target_name = "Missed"
      file_ext = "xlsx" 

    else:
      raise NameError('Project name is not recognized. Fix it in loadData.py file.')
      

    PklFile = f'./Data/{df_name}.pkl'
    df_path = f'./Data/{df_name}.{file_ext}'
        
    if file_ext == 'xlsx':
      df = pd.read_excel(df_path, na_values='?', keep_default_na=True)
    if file_ext == 'csv':
      if df_name == 'cardio_train' or df_name == 'hospital_death' or df_name == 'titanic':
        df = pd.read_csv(df_path, na_values='?', keep_default_na=True ,sep=";")
      else:
        df = pd.read_csv(df_path, na_values='?', keep_default_na=True)
    
    # Set target variable
    df.rename(columns={target_name: "Target"}, inplace=True)
    
    # Remove constant columns
    df.loc[:, (df != df.iloc[0]).any()]

    # Remove id columns (with x_) and columns which are the same as target
    for column in df.columns:
      if 'x_' in column:# or df[column]==df['Target']:
        df.drop(column, axis=1, inplace=True)
    
    # Remove rows with Nan in Target
    df = df.dropna(how='any', subset=['Target'])
    

    # Remove almost constant and similar columns
    columns = list(df.columns)
    columns_to_remove = set()
    nRows = df.shape[0]
    small_difference = max(1, round(nRows*0.001))
    for i in range(len(columns)):
        column = df[columns[i]]
        if column.value_counts().max() > nRows-small_difference:
            columns_to_remove.add(columns[i])
            continue
        for j in range(i + 1, len(columns)):
            other = df[columns[j]]
            if (column != other).sum() < small_difference: 
                columns_to_remove.add(columns[j])
                            
    df.drop(columns_to_remove, axis=1, inplace=True)

    #Specific preprocessing (depends on a dataset)
    # For Covid - none
    # For Bunkruption - none
    # For Miocarda
    if df_name == 'Data_Miocarda':
      alt_targets = ['Outcome_113_Atrial_fibrillation_','Outcome_114_Supraventricular_tachycardia_',
           'Outcome_115_Ventricular_tachycardia_','Outcome_116_Ventricular_fibrillation_',
           'Outcome_117_Third-degree_AV_block_', 'Outcome_118_Pulmonary_edema_', 'Outcome_119_Myocardial_rupture_',
           'Outcome_120_Dressler_syndrome_', 'Outcome_121_Chronic_heart_failure_', 
           'Outcome_122_Relapse_of_the_myocardial_infarction_', 'Outcome_123_Post-infarction_angina_',
           'Outcome_124_Lethal_outcome_']
      alt_targets.remove(target_name)
      df.drop(alt_targets, axis=1, inplace=True)

    if df_name == 'WineQT':
      df['Target'] = df['Target'].apply(lambda x: 0 if x <= 3 else 1)

    if df_name == 'heart_2020':
      df['Target'] = df['Target'].apply(lambda x: 0 if x == 'No' else 1)

    if df_name == 'breast-cancer':
      df['Target'] = df['Target'].apply(lambda x: 0 if x == 'B' else 1)
      df.drop('id', axis=1, inplace=True)

    if df_name == 'heart':
      true_indices_ratio = 0.05
      false_indices_ratio = 0.5

      df['bad_feature'] = np.nan
      true_indices = df.index[df['Target'] == True].tolist()
      bad_feature_true_indices = random.sample(true_indices, k=int(len(true_indices) * true_indices_ratio))
      mask_true = np.full(df.shape[0], False)
      np.put(mask_true, bad_feature_true_indices, True)
      mask_true = pd.Series(mask_true)
      df['bad_feature'] = df['bad_feature'].where(~mask_true, True)

      false_indices = df.index[df['Target'] == False].tolist()
      bad_feature_false_indices = random.sample(false_indices, k=int(len(false_indices) * false_indices_ratio))
      mask_false = np.full(df.shape[0], False)
      np.put(mask_false, bad_feature_false_indices, True)
      mask_false = pd.Series(mask_false)
      df['bad_feature'] = df['bad_feature'].where(~mask_false, False)
      print(df['bad_feature'].value_counts(normalize=True, dropna=False))
      print(df['bad_feature'].value_counts(dropna=False))

    # Change column names
    df.columns = list(map(change_column_name,df.columns))
    
    return df