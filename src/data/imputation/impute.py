import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import os
resolved_path = os.path.abspath('../../models')

print(resolved_path)
print(os.path.exists(resolved_path))
import sys
sys.path.append(resolved_path)
from tune_model import tune

# function to train the imputer
def train_imputer(train, y_train, path_to_save, model, param_space, scoring, normalize, refit):
    # if the data should be normalized
    if normalize:
        train = normalize(train)
    
    print('Fine tuning the imputer...', flush=True)
    # create the imputer
    best_params, best_model = tune(train, y_train , param_space, 
                                   scoring, model, modeltype='reg', 
                                   search_type='grid', n_splits=3, n_repeats=1, 
                                   random_state=1, verbose=True, display_plots=False, 
                                   refit=refit)
    print('Fine tuning completed.', flush=True)
    
    # save the model
    with open(path_to_save, 'wb') as f:
        pkl.dump(best_model, f)

    print('Imputer saved.', flush=True)

    return best_model


def impute(train, test, y_train, model, param_space, 
           scoring, path_to_save=None, normalize=False, 
           retrain_if_exists=False, refit=None, path_data=None):
    
    if refit is None:
        refit = 'mean_test_score'

    # check normalize and retrain_if_exists are booleans
    if not isinstance(normalize, bool):
        raise ValueError('normalize must be a boolean')
    if not isinstance(retrain_if_exists, bool):
        raise ValueError('retrain_if_exists must be a boolean')
    
    # check if the path to save is specified
    if path_to_save is None:
        raise ValueError('path_to_save must be specified')

    # if it is specified not to retrain the model
    if not retrain_if_exists:
        # try to load the model
        try:
            with open(path_to_save, 'rb') as f:
                imputer = pkl.load(f)

            # if the path to the data is specified (for knn imputer it is better to save the data as well)
            if type(path_data) is not type(None):
                # load the pickle file
                with open(path_data, 'rb') as f:
                    imputed_test = pkl.load(f)
                
        # if it does not exist, train the model
        except:
            imputer = train_imputer(train, y_train, path_to_save, model, 
                                    param_space, scoring, normalize, refit)

    # if it is specified to retrain the model
    else:
        imputer = train_imputer(train, y_train, path_to_save, model, 
                                param_space, scoring, normalize, refit)

    # impute the data
    print('Imputing the test data...', flush=True)
    imputed_test = imputer.transform(test)

    if type(path_data) is not type(None):
        # save the imputed data
        with open(path_data, 'wb') as f:
            pkl.dump(imputed_test, f)
    
    print('Imputation completed.', flush=True)

    return imputed_test