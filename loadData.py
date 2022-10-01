import pandas as pd
import os.path as path
import re

def change_column_name(string):
  #remove leading and trailing characters (spaces)
  string = re.sub('^ | $','', string)
  #substitute spaces with _ and leave only letters, digits and _ 
  string = re.sub('\W+','', string.replace(' ','_'))
  return string

def LoadData(project_name, pkl_reload=False):
    
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
      target_name = "cardio"
      file_ext = "csv" 

    else:
      raise NameError('Project name is not recognized. Fix it in loadData.py file.')
      

    PklFile = f'./Data/{df_name}.pkl'
    df_path = f'./Data/{df_name}.{file_ext}'
    
    if not path.exists(PklFile):
        pkl_reload = True
        
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

    # Change column names
    df.columns = list(map(change_column_name,df.columns))
    
    if pkl_reload:
        df.to_pickle(PklFile)
    
    return df