import pandas as pd
import re

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


#P.S. Колонки с малым количеством единиц собираем в колонку 'others', а колонки, где мало нулей просто убираем
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
        string = str(rreplace(string, '_', '∈'))
    elif '<=' not in string:
        string = str(rreplace(string, '_', '='))
        
    return string



def replace_nans(one_hot,column_name):
    list_of_columns = list(one_hot.columns)
    if column_name+'_nan' in list_of_columns:
        list_of_columns.remove(column_name+'_nan')
        one_hot.loc[one_hot[column_name+'_nan']==1, list_of_columns]=None
    return one_hot



def binarize(df, column_name, unique_threshold, q, exceptions_threshold, numerical_binarization, nan_threshold, share_to_drop):
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
                new_column = pd.Series()
                new_column = pd.qcut(x = converted, q=q, duplicates='drop') #NaNs remain as NaNs
                one_hot = one_hot_encoding(new_column, column_name, numerical_binarization='False', share_to_drop=share_to_drop)
                
                if numerical_binarization=='threshold':
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
                if df[column_name].value_counts(normalize=True).max()>share_to_drop:
                    one_hot = one_hot_encoding(df[column_name], column_name, numerical_binarization='range', share_to_drop=share_to_drop)
                    one_hot = replace_nans(one_hot,column_name)
                else:
                    return None
      
        else: #number of expeptions during convertation is large (more than 1%) -> treat as category
            if df[column_name].value_counts(normalize=True).max()>share_to_drop:
                df[column_name] = df[column_name].apply(lambda string: string_preproccesing(str(string)))
                one_hot = one_hot_encoding(df[column_name], column_name, numerical_binarization='range', share_to_drop=share_to_drop)
                one_hot = replace_nans(one_hot,column_name)
            else:
                return None
        one_hot.columns = list(map(add_equal_suffix,one_hot.columns))
    
        return one_hot




def binarize_df(df, boolarize=True, unique_threshold=20, q=20, exceptions_threshold=0.01, numerical_binarization='threshold', nan_threshold = 0.8, share_to_drop=0.05):
    for column in df.columns:
        if column != 'Target':
            print(column)
            binarized = binarize(df, column, unique_threshold, q, exceptions_threshold, numerical_binarization, nan_threshold, share_to_drop)
            if binarized is not None:
                df = df.join(binarized)
                df.drop(column, axis=1, inplace=True)
    if boolarize:
        for i in df.columns:
            df[i] = df[i].apply(lambda x: False if x==0 else True)
    return df