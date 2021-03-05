import preprocessing_functions as pf
import config

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
dataframe = pf.load_data(config.PATH_TO_DATASET)

# divide data set
X_train, X_tests, Y_train, Y_tests = pf.divide_train_test(dataframe, config.TARGET)


# get first letter from cabin variable
X_train['cabin'] = pf.extract_cabin_letter(X_train, 'cabin')


# impute categorical variables
for el in config.CATEGORICAL_VARS:
    X_train[el] = pf.impute_na(X_train, el, replacement='Missing')


# impute numerical variable
for el in config.NUMERICAL_TO_IMPUTE:
    X_train[el + '_NA'] = pf.add_missing_indicator(X_train, el)
    # impute NaNs
    X_train[el] = pf.impute_na(X_train, el, replacement = config.IMPUTATION_DICT[el])


# Group rare labels
for el in config.CATEGORICAL_VARS:
    X_train[el] = pf.remove_rare_labels(X_train, el, config.FREQUENT_LABELS[el])


# encode categorical variables
for el in config.CATEGORICAL_VARS:
    X_train = pf.encode_categorical(X_train, el)


# check all dummies were added
X_train = pf.check_dummy_variables(X_train, config.DUMMY_VARIABLES)


# train scaler and save
sc_X = pf.train_scaler(X_train, config.OUTPUT_SCALER_PATH)


# scale train set
X_train = sc_X.transform(X_train)


# train model and save
pf.train_model(X_train, Y_train, config.OUTPUT_MODEL_PATH)


print('Finished training')