# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import warnings


# ignore warnings
warnings.filterwarnings('ignore')
def tune(X, y, space, scoring, 
         model, modeltype='clf', search_type='grid', n_iter_random=100,
         n_splits=5, n_repeats=3, random_state=1,
         verbose=True, display_plots=False):
    
    # define evaluation
    if modeltype == 'clf':
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    elif modeltype == 'reg':
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
  
    if verbose:
        verbosity = 1

    # define search
    if search_type == 'grid':
        search = GridSearchCV(model, space, scoring=scoring, n_jobs=-1, cv=cv, verbose=verbosity, refit='AUC')
    elif search_type == 'random':
        search = RandomizedSearchCV(model, space, scoring=scoring, n_jobs=-1, cv=cv, n_iter=n_iter_random, verbose=verbosity, refit='AUC')
    
    # execute search
    result = search.fit(X, y)
    
    # plot results
    if display_plots:
        results_df = pd.DataFrame(result.cv_results_)
        for key, values in space.items():
            
            # group the results by the hyperparameter
            param_means = []
            param_stds = []
            for value in values:
                mask = results_df['param_' + key] == value
                param_means.append(np.mean(results_df[mask]['mean_test_score']))
                param_stds.append(np.std(results_df[mask]['mean_test_score']))
            
            # create plot with two subplots side by side
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(key)
            ax[0].plot(values, param_means)
            ax[0].set_title('Mean test scores')
            ax[0].set_xlabel(key)
            ax[0].set_ylabel('mean scores')
            padding = 0.1
            ax[0].set_ylim(max(0, min(param_means) - padding), min(1, max(param_means) + padding))

            ax[1].plot(values, param_stds)
            ax[1].set_title('Mean score std')
            ax[1].set_xlabel(key)
            ax[1].set_ylabel('score std')
            padding = 0.05
            ax[1].set_ylim(max(0, min(param_stds) - padding), min(1, max(param_stds) + padding))

            plt.show()

    # summarize result
    if verbose:
        print('Best Score: %s' % result.best_score_)
        print('Best Hyperparameters:')
        for k, v in result.best_params_.items():
            print('%s: %s' % (k, v))

    # best model
    best_model = result.best_estimator_

    return result.best_params_, best_model