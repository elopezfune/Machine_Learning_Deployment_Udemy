import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    return pd.read_csv(df_path)



def divide_train_test(df, target):
    #copy the dataframe
    df = df.copy()
    
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(df.drop(str(target), axis=1), df[str(target)],
                                                        test_size=0.2,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test
    

def extract_cabin_letter(df, var):
    #copy the dataframe
    df = df.copy()
    
    # captures the first letter
    return df[str(var)].str[0]



def add_missing_indicator(df, var):
    #copy the dataframe
    df = df.copy()
    
    # function adds a binary missing value indicator
    return np.where(df[str(var)].isna(), 1, 0)


    
def impute_na(df, var, replacement):
    #copy the dataframe
    df = df.copy()
    
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    return df[str(var)].fillna(replacement)



def remove_rare_labels(df, var, frequent_labels):
    #copy the dataframe
    df = df.copy()
    
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    return np.where(df[str(var)].isin(frequent_labels), df[str(var)], 'Rare')
    


def encode_categorical(df, var):
    #copy the dataframe
    df = df.copy()
    
    # adds ohe variables and removes original categorical variable
    df = pd.concat([df, pd.get_dummies(df[str(var)], prefix=str(var), drop_first=True)], axis=1)
    df.drop(labels=[str(var)], axis=1, inplace=True)
    return df



def check_dummy_variables(df, dummy_list):
    #copy the dataframe
    df = df.copy()
    
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    missing_values = [el for el in dummy_list if el not in df.columns]
    if len(missing_values) == 0:
        print('Dummy variables are added')
    else:
        for el in missing_values:
            df[str(el)] = 0
    return df
    

def train_scaler(df, output_path):
    #copy the dataframe
    df = df.copy()
    
    # train and save scaler
    sc_X = StandardScaler()
    sc_X = sc_X.fit(df)
    joblib.dump(sc_X, output_path)
    return sc_X
  
    

def scale_features(df, output_path):
    #copy the dataframe
    df = df.copy()
    
    # load scaler and transform data
    sc_X = joblib.load(output_path)
    return sc_X.transform(df)



def train_model(df, target, output_path):
    # copy the dataframe
    df = df.copy()
    
    # define the model
    model = LogisticRegression(C=0.0005, random_state=0)
    # train and save model
    model.fit(df, target)
    joblib.dump(model, output_path)
    return None
 


def predict(df, model):
    #copy the dataframe
    df = df.copy()
    
    # load model and get predictions
    model = joblib.load(model)
    return model.predict(df)