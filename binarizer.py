import pandas as pd
import re
import numpy as np 
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
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
    cols_to_remove = []

    for col in one_hot.columns:
    
        #drop constant columns
        if one_hot[col].nunique()==1:
            one_hot = one_hot.drop(columns=col)
            cols_to_remove.append(col)   
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
                one_hot['others']=one_hot[others_list].sum(axis=1)
                one_hot.loc[one_hot['others']>1, 'others']=1
                one_hot=one_hot.drop(columns=others_list)

                
    #for numerical variables with thresholds       
    elif numerical_binarization=='threshold':
        ones_dict = dict(zip(one_hot.columns,list_of_ones))
        others_list = [key for key, value in ones_dict.items() if value < share_to_drop]
            
        #create 'other' column
        if len(others_list)>1:
            print('United columns:', list(map(add_equal_suffix,others_list)))
            one_hot['others']=one_hot[others_list].sum(axis=1)
            one_hot.loc[one_hot['others']>1, 'others']=1
            one_hot=one_hot.drop(columns=others_list)
    
    #drop columns which have too many ones
    for col in one_hot.columns:
        if one_hot[col].value_counts(normalize=True)[0]<share_to_drop:
            one_hot = one_hot.drop(columns=col)
            cols_to_remove.append(col)       
                
    return one_hot, cols_to_remove, others_list



def one_hot_encoding(column, column_name, numerical_binarization, share_to_drop):
    #if there is no nans, do not create a na column
    dummy_na = True
    if len(column[column.isna()])==0:
        dummy_na = False
    #check if the column have only two different unique values (then create just one binary column)
    if column.dropna().nunique()==2:
        one_hot = pd.get_dummies(column, dummy_na=dummy_na, drop_first=True) #creates NaN-column
        
    else:
        one_hot = pd.get_dummies(column, dummy_na=dummy_na) #creates NaN-column
        #drop duplicated columns
        one_hot = one_hot.T.drop_duplicates().T 
    
        if numerical_binarization=='range' or numerical_binarization=='threshold':
            one_hot, _, _ = drop_unbalanced(one_hot, column_name, numerical_binarization, share_to_drop) 
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
    try:
        string = '<=' + str(string.right)
    except:
        string = str(string)
        #function replace the last occurance of a character
        def rreplace(s, old, new):
            return (s[::-1].replace(old[::-1],new[::-1], 1))[::-1]

        if '(' in string and ']' in string:
            string = str(rreplace(string, '_', '$\in$'))
        elif '<=' not in string:
            string = str(rreplace(string, '_', '='))

    return string



def replace_nans(one_hot,column_name, df):
    list_of_columns = list(one_hot.columns)
    if column_name+'_nan' in list_of_columns:
        list_of_columns.remove(column_name+'_nan')
    one_hot.loc[df[column_name]=='nan', list_of_columns]=None
    return one_hot



def binarize(df, column_name, unique_threshold, q, exceptions_threshold, numerical_binarization, nan_threshold, share_to_drop, create_nan_features, dict_strategies, dict_one_hot_values):
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
                new_column = pd.qcut(x = converted, q=q, duplicates='drop') #NaNs remain as NaNs
                dict_strategies[column_name] = 'range'
                one_hot = one_hot_encoding(new_column, column_name, numerical_binarization='False', share_to_drop=share_to_drop)             

                if numerical_binarization=='threshold':
                    
                    dict_strategies[column_name] = 'threshold'
                    #rename columns as 'x<=a'
                    if 'nan' in one_hot.columns:
                        nan_col = one_hot.pop('nan')
                    list_of_columns = list(one_hot.columns)
                    for column_i in range(1,len(list_of_columns)+1):
                        one_hot.iloc[(one_hot[list_of_columns[column_i-1]]==1),column_i-1:]=1
                    try:
                        one_hot['nan'] = nan_col
                    except:
                        pass
          
                df[column_name] = df[column_name].apply(lambda string: string_preproccesing(str(string)))
                one_hot, cols_to_remove, others_list = drop_unbalanced(one_hot, column_name, numerical_binarization, share_to_drop)

                def interval_right_bound(interval):
                    try:
                        return interval.right
                    except:
                        return interval

                def interval_bounds(interval):
                    try:
                        return (interval.left, interval.right)
                    except:
                        return interval


                if numerical_binarization=='threshold':
                    one_hot_cols = list(one_hot.columns)
                    print(one_hot_cols)
                    if not type(one_hot_cols[-1])==float:
                        one_hot_cols[-1]=float('inf')
                    else:
                        one_hot_cols[-2]=float('inf')
                    one_hot.columns = one_hot_cols
                    print(one_hot.columns)
                    dict_one_hot_values[column_name] = list(map(interval_right_bound, list(one_hot.columns)))
                    #print(dict_one_hot_values[column_name])

                elif numerical_binarization=='range':
                    one_hot_cols = list(one_hot.columns)
                    print(one_hot_cols)
                    one_hot_cols[0]=(float('-inf'), one_hot.columns[0].right)
                    if not type(one_hot_cols[-1])==float:
                        one_hot_cols[-1]=(one_hot.columns[-1].left, float('inf'))
                    else:
                        one_hot_cols[-2]=(one_hot.columns[-2].left, float('inf'))
                    one_hot.columns = one_hot_cols
                    print(one_hot.columns)
                    dict_one_hot_values[column_name] = list(map(interval_bounds, list(one_hot.columns)))
                    #print(dict_one_hot_values[column_name])

                try:
                    dict_one_hot_values[column_name][dict_one_hot_values[column_name].index(np.nan)]='nan'
                except:
                    pass

                one_hot = one_hot.add_prefix(column_name + '_')
                if numerical_binarization=='threshold':
                    one_hot.columns = list(map(rename_columns,one_hot.columns))
                one_hot = replace_nans(one_hot,column_name, df)

                for col in cols_to_remove:
                    try:
                        dict_one_hot_values[column_name].remove(col)
                    except:
                        pass

                for col in others_list:
                    try:
                        dict_one_hot_values[column_name].remove(col)
                    except:
                        pass
          
                    # if numerical_binarization=='threshold': 
                    #     dict_one_hot_values[column_name][-1] = float('inf')

                    # elif numerical_binarization=='range':
                    #     dict_one_hot_values[column_name][0] = (float('-inf'), dict_one_hot_values[column_name][0][1])
                    #     dict_one_hot_values[column_name][-1] = (dict_one_hot_values[column_name][-1][0],  float('inf'))

                print(dict_one_hot_values[column_name])
                
        
            #number of unique values is less than 20  -> treat as category
            else: 

                if df[column_name].value_counts(normalize=True).max()>share_to_drop:
                    dict_strategies[column_name] = 'category_numeric'

                    one_hot = one_hot_encoding(df[column_name], column_name, numerical_binarization='range', share_to_drop=share_to_drop)
                    dict_one_hot_values[column_name] = list(one_hot.columns)
                    one_hot = one_hot.add_prefix(column_name + '_')

                    one_hot = replace_nans(one_hot,column_name,df)
                
                else:
                    return None
      
        else: #number of expeptions during convertation is large (more than 1%) -> treat as category

            if df[column_name].value_counts(normalize=True).max()>share_to_drop:
                dict_strategies[column_name] = 'category'
                df[column_name] = df[column_name].apply(lambda string: string_preproccesing(str(string)))
                one_hot = one_hot_encoding(df[column_name], column_name, numerical_binarization='range', share_to_drop=share_to_drop)
                dict_one_hot_values[column_name] = list(one_hot.columns)
                one_hot = one_hot.add_prefix(column_name + '_')
                one_hot = replace_nans(one_hot,column_name, df)
            else:
                return None

        one_hot.columns = list(map(add_equal_suffix,one_hot.columns))
    
        # removing nan columns
        if not create_nan_features:
            for col in one_hot.columns:
                if '=nan' in col:
                    one_hot = one_hot.drop(columns = [col])
                    dict_one_hot_values[column_name].remove('nan')
         
        return one_hot, dict_strategies, dict_one_hot_values

    else:
        return None, dict_strategies, dict_one_hot_values

def boolarize(value):
    if value==1:
        return True
    elif value==0:
        return False
    else:
        return np.nan



def binarize_df(df, unique_threshold=20, q=20, exceptions_threshold=0.01, numerical_binarization='threshold', nan_threshold = 0.8, share_to_drop=0.05, create_nan_features=True):
    dict_strategies = {}
    dict_one_hot_values = {}


    for column in df.columns:
        if column != 'Target':
            print(column)
            binarized, dict_strategies, dict_one_hot_values = binarize(df, column, unique_threshold, q, exceptions_threshold, numerical_binarization, nan_threshold, share_to_drop, create_nan_features, dict_strategies, dict_one_hot_values)
            if binarized is not None:
                df = df.join(binarized)
                df.drop(column, axis=1, inplace=True)
            else:
                df.drop(column, axis=1, inplace=True)
    
    # convert to boolean type
    for col in df.columns:
        df[col] = df[col].apply(boolarize)
    
    return df, dict_strategies, dict_one_hot_values


### Сюда должен подаваться датасет (test_df) после loadData
def binarizer_predict(test_df, column_name, dict_strategies, dict_one_hot_values):
    one_hot = pd.DataFrame()
    strategy = dict_strategies[column_name]
    list_of_cols = dict_one_hot_values[column_name]

    if strategy=='threshold' or strategy=='range' or strategy=='category_numeric':
        converted = pd.to_numeric(test_df[column_name].copy(), downcast='float', errors='coerce')
        
        if strategy=='threshold':
            for col in list_of_cols:
                if col != 'nan' and col!='others':
                    name = column_name + '<=' + str(col)
                    #parsed_value = col
                    one_hot[name] = converted<=col
                elif col=='nan':
                    name = column_name + '=nan'
                    one_hot[name] = converted.isna()
                    

        elif strategy=='range':
            for col in list_of_cols:
                if col!='nan' and col!='others':
                    lower_bound = col[0]
                    upper_bound = col[1]
                    name = column_name + '$\in$(' + str(lower_bound) +', ' + str(upper_bound) + ']'
                    # print(test_df[column_name])
                    one_hot[name] = (lower_bound<converted) & (converted<=upper_bound)
                elif col=='nan':
                    name = column_name + '=nan'
                    one_hot[name] = converted.isna()

        else:
            for col in list_of_cols:
                if col!='nan' and col!='others':
                    name = column_name + '=' + str(col)
                    one_hot[name] = test_df[column_name]==col
                elif col=='nan':
                    name = column_name + '=nan'
                    one_hot[name] = test_df[column_name].isna()
            
    elif strategy=='category':
        test_df[column_name] = test_df[column_name].apply(lambda string: string_preproccesing(str(string)))
        for col in list_of_cols:
            if col!='nan' and col!='others':
                name = column_name + '=' + col
                one_hot[name] = test_df[column_name]==col
            elif col=='nan':
                name = column_name + '=nan'
                one_hot[name] = test_df[column_name].isna()
    
    def replace_others(one_hot,column_name):
        name = column_name + '=others'
        one_hot[name] = False
        one_hot.loc[(one_hot.apply(lambda row: True if True not in list(row) else False, axis=1)), name]= True
        return one_hot
    
    if 'others' in list_of_cols:
        one_hot = replace_others(one_hot,column_name)
        
    def replace_nans(one_hot,column_name, test_df):
        list_of_columns = list(one_hot.columns)
        if column_name+'=nan' in list_of_columns:
            list_of_columns.remove(column_name+'=nan')
        one_hot.loc[test_df[column_name]=='nan', list_of_columns]=np.nan
        return one_hot

    one_hot = replace_nans(one_hot,column_name, test_df)
    return one_hot

def binarize_test(df, dict_strategies, dict_one_hot_values):
    for column in df.columns:
        if column != 'Target' and column in dict_strategies.keys():
            binarized_ = binarizer_predict(df, column, dict_strategies, dict_one_hot_values)
            df = df.join(binarized_)
            df.drop(column, axis=1, inplace=True)
        elif column == 'Target':
            df.loc[(df['Target']==1), 'Target'] = True
            df.loc[(df['Target']==0), 'Target'] = False
        else:
            df.drop(column, axis=1, inplace=True)
    return df