import preprocessing_functions as pf
import config

# =========== scoring pipeline =========

# impute categorical variables
def predict(data):
    
    # extract first letter from cabin
    data['cabin'] = pf.extract_cabin_letter(data, 'cabin')

    # impute NA categorical
    for el in config.CATEGORICAL_VARS:
        data[el] = pf.impute_na(data, el, replacement='Missing')
    
    
    # impute NA numerical
    for el in config.NUMERICAL_TO_IMPUTE:
        
        # add missing indicator first
        data[el + '_NA'] = pf.add_missing_indicator(data, el)
        
        # impute NA
        data[el] = pf.impute_na(data, el, replacement = config.IMPUTATION_DICT[el])
    
    
    # Group rare labels
    for el in config.CATEGORICAL_VARS:
        data[el] = pf.remove_rare_labels(data, el, config.FREQUENT_LABELS[el])

    
    # encode variables
    for el in config.CATEGORICAL_VARS:
        data = pf.encode_categorical(data, el)

        
        
    # check all dummies were added
    data = pf.check_dummy_variables(data, config.DUMMY_VARIABLES)

    
    # scale variables
    data = pf.scale_features(data, config.OUTPUT_SCALER_PATH)

    
    # make predictions
    predictions = pf.predict(data, config.OUTPUT_MODEL_PATH)

    return predictions

# ======================================
    
# small test that scripts are working ok
    
if __name__ == '__main__':
        
    from sklearn.metrics import accuracy_score    
    import warnings
    warnings.simplefilter(action='ignore')
    
    # Load data
    data = pf.load_data(config.PATH_TO_DATASET)
    
    X_train, X_test, y_train, y_test = pf.divide_train_test(data,
                                                            config.TARGET)
    
    pred = predict(X_test)
    
    # evaluate
    # if your code reprodues the notebook, your output should be:
    # test accuracy: 0.6832
    print('test accuracy: {}'.format(accuracy_score(y_test, pred)))
    print()
        