import pandas as pd
import re
import numpy as np 

#Parameters description

#unique_threshold - maximal number of unique values to consider numerical variable as category
#q - number of quantiles to split numerical variable
#exceptions_threshold - max % of exeptions allowed while converting a variable into numeric to treat it as numeric
#numerical_binarization - type of numerical binarization (range or threshold)
#nan_threshold - max % of missing values allowed to process the variable
#share_to_drop - max % of zeros allowed for a binarized column OR joint % of ones for the the most unballanced columns (with the least number of ones) which are joined together into 'other' category.



def string_preproccesing(string):
    string = re.sub('\W+','', string.lower().replace('_',''))
    return string


def drop_unbalanced(one_hot, column_name, numerical_binarization, share_to_drop):
    
    #create a list for the share of ones for each binarized column of one_hot dataset
    list_of_ones = []
    for col in one_hot.columns:
            
        #drop constant columns
        if one_hot[col].nunique()==1:
            one_hot = one_hot.drop(columns=col)    
        else:
            list_of_ones.append(one_hot[col].value_counts(normalize=True)[1])

    #for categorical variables and for numerical variables with range binarization           
    if numerical_binarization=='range':
        
        #if the column was binary -> pass the processing
        if len(one_hot.columns)==1:
            pass
        else:
            
            #create an ordered dict with col names as keys and share of ones as values (in descending order)
            #and define columns which jointly have less then share_to_drop ones
            ones_dict = sorted(dict(zip(one_hot.columns,list_of_ones)).items(), key=lambda x: x[1], reverse=True)
            col_sum = 0
            i=0
            while col_sum<=1-share_to_drop and i<len(ones_dict): #and i<len(ones_dict):
                col_sum+=ones_dict[i][1]
                i+=1
            others_list =[]
            if i!=len(ones_dict)-1:
                for m in range(i,len(ones_dict)): 
                    others_list.append(ones_dict[m][0])
                    
            #create 'other' column        
            if len(others_list)>1:
                print('United columns:', list(map(add_equal_suffix,others_list)))
                one_hot[column_name+'_others']=one_hot[others_list].sum(axis=1)
                one_hot.loc[one_hot[column_name+'_others']>1, column_name+'_others']=1
                one_hot=one_hot.drop(columns=others_list)

                
    #for numerical variables with thresholds       
    elif numerical_binarization=='threshold':
        ones_dict = dict(zip(one_hot.columns,list_of_ones))
        others_list = [key for key, value in ones_dict.items() if value < share_to_drop]
            
        #create 'other' column
        if len(others_list)>1:
            print('United columns:', others_list)
            one_hot[column_name+'_others']=one_hot[others_list].sum(axis=1)
            one_hot.loc[one_hot[column_name+'_others']>1, column_name+'_others']=1
            one_hot=one_hot.drop(columns=others_list)
    
    #drop columns which have too many ones
    for col in one_hot.columns:
        if one_hot[col].value_counts(normalize=True)[0]<share_to_drop:
            one_hot = one_hot.drop(columns=col)      
                
    return one_hot



def one_hot_encoding(column, column_name, numerical_binarization, share_to_drop):
    #if there is no nans, do not create a na column
    dummy_na = True
    if len(column[column.isna()])==0:
        dummy_na = False
    #check if the column have only two different unique values (then create just one binary column)
    if column.dropna().nunique()==2:
        one_hot = pd.get_dummies(column, prefix=column_name, dummy_na=dummy_na, drop_first=True) #creates NaN-column
        
    else:
        one_hot = pd.get_dummies(column, prefix=column_name, dummy_na=dummy_na) #creates NaN-column
        #drop duplicated columns
        one_hot = one_hot.T.drop_duplicates().T 
    
        if numerical_binarization=='range' or numerical_binarization=='threshold':
            one_hot = drop_unbalanced(one_hot, column_name, numerical_binarization, share_to_drop) 
        else:
            pass
    
    return one_hot

   
    
def rename_columns(string):
    if '(' in string and ']' in string:
        string = str(string.split('_(')[0]) + str('<=')+ str(string.split(', ')[1].replace(']',''))
    else:
        pass
    return string  


def add_equal_suffix(string):
    #function replace the last occurance of a character
    def rreplace(s, old, new):
        return (s[::-1].replace(old[::-1],new[::-1], 1))[::-1]
    
    if '(' in string and ']' in string:
        string = str(rreplace(string, '_', '$\in$'))
    elif '<=' not in string:
        string = str(rreplace(string, '_', '='))
        
    return string



def replace_nans(one_hot,column_name):
    list_of_columns = list(one_hot.columns)
    if column_name+'_nan' in list_of_columns:
        list_of_columns.remove(column_name+'_nan')
        one_hot.loc[one_hot[column_name+'_nan']==1, list_of_columns]=None
    return one_hot



def binarize(df, column_name, unique_threshold, q, exceptions_threshold, numerical_binarization, nan_threshold, share_to_drop, create_nan_features, dict_strategies):
    num_NA = len(df[df[column_name].isna()])
    nan_share = num_NA/len(df[column_name])
    
    #consider columns with not a lot of missings
    if nan_share<nan_threshold: 
        
        #try to convert into numeric
        converted = pd.to_numeric(df[column_name].copy(), downcast='float', errors='coerce') #exeptions are replaced with NaN
        num_NA_converted = len(converted[converted.isna()])
        
        #number of expeptions during convertation is small (less than 1%) -> column contains numbers 
        if ((num_NA_converted-num_NA)<exceptions_threshold*len(df[column_name])):
            
            #number of unique values is more than threshold (+1 for NaN) -> treat as numeric
            if (converted.nunique()>unique_threshold+1):
                dict_strategies[column_name] = 'range'
                new_column = pd.qcut(x = converted, q=q, duplicates='drop') #NaNs remain as NaNs
                one_hot = one_hot_encoding(new_column, column_name, numerical_binarization='False', share_to_drop=share_to_drop)
                
                if numerical_binarization=='threshold':
                    dict_strategies[column_name] = 'threshold'
                    #rename columns as 'x<=a'
                    one_hot.columns = list(map(rename_columns,one_hot.columns))
                    if column_name+'_nan' in one_hot.columns:
                        nan_col = one_hot.pop(column_name+'_nan')
                    list_of_columns = list(one_hot.columns)
                    for column_i in range(1,len(list_of_columns)+1):
                        one_hot.iloc[(one_hot[list_of_columns[column_i-1]]==1),column_i-1:]=1
                    try:
                        one_hot[column_name+'_nan'] = nan_col
                    except:
                        pass
          
                df[column_name] = df[column_name].apply(lambda string: string_preproccesing(str(string)))
                one_hot = drop_unbalanced(one_hot, column_name, numerical_binarization, share_to_drop)
                one_hot = replace_nans(one_hot,column_name)
        
            #number of unique values is less than 20  -> treat as category
            else: 
                dict_strategies[column_name] = 'category_numeric'
                if df[column_name].value_counts(normalize=True).max()>share_to_drop:
                    one_hot = one_hot_encoding(df[column_name], column_name, numerical_binarization='range', share_to_drop=share_to_drop)
                    one_hot = replace_nans(one_hot,column_name)
                else:
                    return None
      
        else: #number of expeptions during convertation is large (more than 1%) -> treat as category
            dict_strategies[column_name] = 'category'
            if df[column_name].value_counts(normalize=True).max()>share_to_drop:
                df[column_name] = df[column_name].apply(lambda string: string_preproccesing(str(string)))
                one_hot = one_hot_encoding(df[column_name], column_name, numerical_binarization='range', share_to_drop=share_to_drop)
                one_hot = replace_nans(one_hot,column_name)
            else:
                return None
        one_hot.columns = list(map(add_equal_suffix,one_hot.columns))
    
        # removing nan columns
        if not create_nan_features:
            for col in one_hot.columns:
                if '=nan' in col:
                    one_hot = one_hot.drop(columns = [col])
         

        return one_hot

def boolarize(value):
    if value==1:
        return True
    elif value==0:
        return False
    else:
        return np.nan



def binarize_df(df, unique_threshold=20, q=20, exceptions_threshold=0.01, numerical_binarization='threshold', nan_threshold = 0.8, share_to_drop=0.05, create_nan_features=True):
    dict_strategies = {}
    dict_one_hot_cols = {}

    for column in df.columns:
        if column != 'Target':
            print(column)
            binarized = binarize(df, column, unique_threshold, q, exceptions_threshold, numerical_binarization, nan_threshold, share_to_drop, create_nan_features, dict_strategies)
            if binarized is not None:
                df = df.join(binarized)
                df.drop(column, axis=1, inplace=True)
                dict_one_hot_cols[column] = list(binarized.columns)
    
    # convert to boolean type
    for col in df.columns:
        df[col] = df[col].apply(boolarize)
    print(dict_one_hot_cols)
    print(dict_strategies)
    return df, dict_one_hot_cols, dict_strategies


### Сюда должен подаваться датасет (test_df) после loadData
### пример того, как выглядят словари
# dict_one_hot_cols = {'Pclass': ['Pclass=1', 'Pclass=2', 'Pclass=3'], 'Sex': ['Sex=male'], 'Age': ['Age$\\in$(0.419, 4.0]', 'Age$\\in$(4.0, 14.0]', 'Age$\\in$(14.0, 17.0]', 'Age$\\in$(17.0, 19.0]', 'Age$\\in$(19.0, 20.125]', 'Age$\\in$(20.125, 22.0]', 'Age$\\in$(22.0, 24.0]', 'Age$\\in$(24.0, 25.0]', 'Age$\\in$(25.0, 27.0]', 'Age$\\in$(27.0, 28.0]', 'Age$\\in$(28.0, 30.0]', 'Age$\\in$(30.0, 31.8]', 'Age$\\in$(31.8, 34.0]', 'Age$\\in$(34.0, 36.0]', 'Age$\\in$(36.0, 38.0]', 'Age$\\in$(38.0, 41.0]', 'Age$\\in$(41.0, 45.0]', 'Age$\\in$(45.0, 50.0]', 'Age$\\in$(50.0, 56.0]', 'Age$\\in$(56.0, 80.0]', 'Age=nan'], 'SibSp': ['SibSp=0', 'SibSp=1', 'SibSp=2', 'SibSp=3', 'SibSp=4', 'SibSp=5', 'SibSp=8'], 'Parch': ['Parch=0', 'Parch=1', 'Parch=2', 'Parch=3', 'Parch=4', 'Parch=5', 'Parch=6'], 'Fare': ['Fare$\\in$(-0.001, 7.225]', 'Fare$\\in$(7.225, 7.55]', 'Fare$\\in$(7.55, 7.75]', 'Fare$\\in$(7.75, 7.854]', 'Fare$\\in$(7.854, 7.91]', 'Fare$\\in$(7.91, 8.05]', 'Fare$\\in$(8.05, 9.0]', 'Fare$\\in$(9.0, 10.5]', 'Fare$\\in$(10.5, 13.0]', 'Fare$\\in$(13.0, 14.454]', 'Fare$\\in$(14.454, 16.1]', 'Fare$\\in$(16.1, 21.679]', 'Fare$\\in$(21.679, 26.0]', 'Fare$\\in$(26.0, 27.0]', 'Fare$\\in$(27.0, 31.0]', 'Fare$\\in$(31.0, 39.688]', 'Fare$\\in$(39.688, 56.496]', 'Fare$\\in$(56.496, 77.958]', 'Fare$\\in$(77.958, 112.079]', 'Fare$\\in$(112.079, 512.329]'], 'Cabin': ['Cabin=a10', 'Cabin=a14', 'Cabin=a16', 'Cabin=a19', 'Cabin=a20', 'Cabin=a23', 'Cabin=a24', 'Cabin=a26', 'Cabin=a31', 'Cabin=a32', 'Cabin=a34', 'Cabin=a36', 'Cabin=a5', 'Cabin=a6', 'Cabin=a7', 'Cabin=b101', 'Cabin=b102', 'Cabin=b18', 'Cabin=b19', 'Cabin=b20', 'Cabin=b22', 'Cabin=b28', 'Cabin=b3', 'Cabin=b30', 'Cabin=b35', 'Cabin=b37', 'Cabin=b38', 'Cabin=b39', 'Cabin=b4', 'Cabin=b41', 'Cabin=b42', 'Cabin=b49', 'Cabin=b5', 'Cabin=b50', 'Cabin=b51b53b55', 'Cabin=b57b59b63b66', 'Cabin=b58b60', 'Cabin=b69', 'Cabin=b71', 'Cabin=b73', 'Cabin=b77', 'Cabin=b78', 'Cabin=b79', 'Cabin=b80', 'Cabin=b82b84', 'Cabin=b86', 'Cabin=b94', 'Cabin=b96b98', 'Cabin=c101', 'Cabin=c103', 'Cabin=c104', 'Cabin=c106', 'Cabin=c110', 'Cabin=c111', 'Cabin=c118', 'Cabin=c123', 'Cabin=c124', 'Cabin=c125', 'Cabin=c126', 'Cabin=c128', 'Cabin=c148', 'Cabin=c2', 'Cabin=c22c26', 'Cabin=c23c25c27', 'Cabin=c30', 'Cabin=c32', 'Cabin=c45', 'Cabin=c46', 'Cabin=c47', 'Cabin=c49', 'Cabin=c50', 'Cabin=c52', 'Cabin=c54', 'Cabin=c62c64', 'Cabin=c65', 'Cabin=c68', 'Cabin=c7', 'Cabin=c70', 'Cabin=c78', 'Cabin=c82', 'Cabin=c83', 'Cabin=c85', 'Cabin=c86', 'Cabin=c87', 'Cabin=c90', 'Cabin=c91', 'Cabin=c92', 'Cabin=c93', 'Cabin=c95', 'Cabin=c99', 'Cabin=d', 'Cabin=d10d12', 'Cabin=d11', 'Cabin=d15', 'Cabin=d17', 'Cabin=d19', 'Cabin=d20', 'Cabin=d21', 'Cabin=d26', 'Cabin=d28', 'Cabin=d30', 'Cabin=d33', 'Cabin=d35', 'Cabin=d36', 'Cabin=d37', 'Cabin=d45', 'Cabin=d46', 'Cabin=d47', 'Cabin=d48', 'Cabin=d49', 'Cabin=d50', 'Cabin=d56', 'Cabin=d6', 'Cabin=d7', 'Cabin=d9', 'Cabin=e10', 'Cabin=e101', 'Cabin=e12', 'Cabin=e121', 'Cabin=e17', 'Cabin=e24', 'Cabin=e25', 'Cabin=e31', 'Cabin=e33', 'Cabin=e34', 'Cabin=e36', 'Cabin=e38', 'Cabin=e40', 'Cabin=e44', 'Cabin=e46', 'Cabin=e49', 'Cabin=e50', 'Cabin=e58', 'Cabin=e63', 'Cabin=e67', 'Cabin=e68', 'Cabin=e77', 'Cabin=e8', 'Cabin=f2', 'Cabin=f33', 'Cabin=f4', 'Cabin=fg73', 'Cabin=g6', 'Cabin=nan', 'Cabin=others'], 'Embarked': ['Embarked=c', 'Embarked=nan', 'Embarked=q', 'Embarked=s']}
# dict_strategies = {'Pclass': 'category_numeric', 'Sex': 'category', 'Age': 'range', 'SibSp': 'category_numeric', 'Parch': 'category_numeric', 'Fare': 'range', 'Cabin': 'category', 'Embarked': 'category'}

def binarizer_predict(test_df, column_name, dict_strategies, dict_one_hot_cols):
    one_hot = pd.DataFrame()
    strategy = dict_strategies[column_name]
    list_of_cols = dict_one_hot_cols[column_name]

    if strategy=='threshold' or strategy=='range' or strategy=='category_numeric':
        converted = pd.to_numeric(test_df[column_name].copy(), downcast='float', errors='coerce')
        
        if strategy=='threshold':
            for col in list_of_cols:
                if '=nan' not in col:
                    parsed_value = float(col[col.find('=')+1:])
                    one_hot[col] = test_df[column_name]<=parsed_value
                else:
                    one_hot[col] = test_df[column_name].isna()

        elif strategy=='range':
            for col in list_of_cols:
                if '=nan' not in col:
                    lower_bound = float(col[col.find('(')+1:col.find(',')])
                    upper_bound = float(col[col.find(',')+2:-1]) 
                    one_hot[col] = (lower_bound<test_df[column_name]) & (test_df[column_name]<=upper_bound)
                else:
                    one_hot[col] = test_df[column_name].isna()

        else:
            for col in list_of_cols:
                if '=nan' not in col:
                    parsed_value = float(col[col.find('=')+1:])
                    one_hot[col] = test_df[column_name]==parsed_value
                else:
                    one_hot[col] = test_df[column_name].isna()
            
    elif strategy=='category':
        for col in list_of_cols:
            if '=nan' not in col:
                parsed_value = str(col[col.find('=')+1:])
                one_hot[col] = test_df[column_name]==parsed_value
            else:
                one_hot[col] = test_df[column_name].isna()
    
    def replace_nans(one_hot,column_name):
        list_of_columns = list(one_hot.columns)
        if column_name+'=nan' in list_of_columns:
            list_of_columns.remove(column_name+'=nan')
            one_hot.loc[one_hot[column_name+'=nan']==1, list_of_columns]=None
        return one_hot

    one_hot = replace_nans(one_hot,column_name)
    return one_hot

def binarize_test(df, dict_strategies, dict_one_hot_cols):
    for column in df.columns:
        if column != 'Target' and column in dict_strategies.keys():
            binarized_ = binarizer_predict(df, column, dict_strategies, dict_one_hot_cols)
            df = df.join(binarized_)
            df.drop(column, axis=1, inplace=True)
    return df