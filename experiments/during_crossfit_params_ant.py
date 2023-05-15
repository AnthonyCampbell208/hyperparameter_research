import csv
import warnings
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import utils
import pickle
import pandas as pd
import numpy as np
import logging
import argparse
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.preprocessing
from econml.dml import CausalForestDML, KernelDML, LinearDML, SparseLinearDML
from econml.dr import DRLearner
from sklearn.base import BaseEstimator, is_regressor
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (ARDRegression, BayesianRidge, ElasticNet,
                                  ElasticNetCV, Lars, Lasso, LassoLars,
                                  LinearRegression, LogisticRegression,
                                  LogisticRegressionCV,
                                  OrthogonalMatchingPursuit, Ridge)
from sklearn.model_selection import (BaseCrossValidator, GridSearchCV, KFold,
                                     RandomizedSearchCV, StratifiedKFold,
                                     check_cv, train_test_split)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler,
                                   PolynomialFeatures, RobustScaler,
                                   StandardScaler)
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from itertools import product
from sklearn.metrics import f1_score, mean_squared_error
import wandb
import sys
from sklearn.preprocessing import StandardScaler
import os
import pdb
from utils import *
import time
<< << << < HEAD

sys.path.insert(
    0, '/Users/anthonycampbell/miniforge3/pkgs/econml-0.13.1-py39h533cade_0/lib/python3.9/site-packages/')

k = 2


if __name__ == "__main__":

    classifiers = [RandomForestClassifier(),
                   LogisticRegressionCV(),
                   ]

    regressors = [RandomForestRegressor(),
                  ElasticNetCV(),
                  ]

    # wandb.init(project="cs696ds-econml", config={
    #     "causal_estimators": ci_estimators,
    #     "classifiers": classifiers,
    #     "regressors": regressors,
    # })
    # config = wandb.config
    np.random.seed(42)
    # data_dict = {}
    data_dict = {}
    data_dict = {'acic': None, 'twin': load_twin(), 'ihdp': load_ihdp()}
    # pdb.set_trace()
    for key in data_dict:
        if key == 'acic':
            iterator = load_acic()
            for acic_data in iterator:
                data, X, T, Y, true_ite, true_ATE, true_ATE_stderr, is_discrete, file_name = acic_data
                scaler = StandardScaler()
                x_scaled = scaler.fit_transform(X)
                results_file = f'results/{key}_during_crossfit_params.csv'
                already_loaded_file = False
                if os.path.exists(results_file):
                    results_df = pd.read_csv(results_file, index_col=0)
                    already_loaded_file = True
                    all_results = results_df.to_dict('records')
                else:
                    all_results = []
                    results_df = pd.DataFrame()
                my_list = classifiers if is_discrete else regressors
                mt_list = classifiers
                i = 0
                kf = KFold(n_splits=2, shuffle=True)
                for fold, (train_index, test_index) in enumerate(kf.split(X)):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    T_train, T_test = T.iloc[train_index], T.iloc[test_index]
                    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
                    for model_y in my_list:
                        count = 0
                        for model_t in mt_list:
                            try:
                                # pdb.set_trace()
                                if os.path.exists(results_file):
                                    # pdb.set_trace()
                                    exists = results_df[
                                        (results_df["model_y"] ==
                                            model_y.__class__.__name__)
                                        & (results_df["model_t"] == model_t.__class__.__name__)
                                        & (results_df['file_name'] == file_name)
                                        & (results_df['fold'] == fold)
                                    ].any().any()
                                    if exists:
                                        print(
                                            f"Skipping model_y: {model_y}, model_t: {model_t}")
                                        continue
                                params_model_y = select_classification_hyperparameters(
                                    model_y) if is_discrete else select_regression_hyperparameters(model_y)
                                params_model_t = select_classification_hyperparameters(
                                    model_t)
                                grid_search_t = GridSearchCV(
                                    model_t, params_model_t, cv=3)
                                grid_search_t.fit(X_train, Y_train)
                                best_params_t = grid_search_t.best_params_

                                grid_search_y = GridSearchCV(
                                    model_y, params_model_y, cv=3)
                                grid_search_y.fit(X_train, Y_train)
                                best_params_y = grid_search_y.best_params_

                                model_t.set_params(**best_params_t)
                                model_y.set_params(**best_params_y)

                                temp_results = {
                                    'model_t': model_t.__class__.__name__, 'model_y': model_y.__class__.__name__, 'fold': fold, 'file_name': file_name}

                                start = time.time()
                                model_t.fit(X_train, T_train)
                                end = time.time()
                                temp_results['time_model_t'] = end - start

                                start = time.time()
                                model_y.fit(X_train, Y_train)
                                end = time.time()
                                temp_results['time_model_y'] = end - start

                                T_pred = model_t.predict(X_train)
                                temp_results['f1_score_model_t'] = f1_score(
                                    T_train, T_pred, average='weighted')  # or 'binary' for binary classification

                                Y_pred = model_y.predict(X_train)

                                if is_discrete:
                                    temp_results['f1_score_model_y'] = f1_score(
                                        Y_train, Y_pred, average='weighted')  # or 'binary' for binary classification
                                else:
                                    temp_results['mse_model_y'] = mean_squared_error(
                                        Y_train, Y_pred)

                                all_results.append(temp_results)
                                results_df = pd.DataFrame(all_results)
                                results_df.to_csv(
                                    f'results/{key}_during_crossfit_params.csv')
                                print(
                                    f"Completed running model_y: {model_y}, model_t: {model_t}")
                            except Exception as e:
                                print(
                                    f"Error occurred while running {model_y}-{model_t}  {str(e)}")
                            i += 1
                            count += 1
                results_df.to_csv(f"results/{key}_during_crossfit_params.csv")
        else:
            data, X, T, Y, true_ite, true_ATE, true_ATE_stderr, is_discrete = data_dict[key]
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(X)
            results_file = f'results/{key}_during_crossfit_params.csv'
            already_loaded_file = False
            if os.path.exists(results_file):
                results_df = pd.read_csv(results_file, index_col=0)
                already_loaded_file = True
                all_results = results_df.to_dict('records')
            else:
                all_results = []
                results_df = pd.DataFrame()
            my_list = regressors
            mt_list = classifiers if is_discrete else regressors
            i = 0

            kf = KFold(n_splits=2, shuffle=True)
            for fold, (train_index, test_index) in enumerate(kf.split(X)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                T_train, T_test = T.iloc[train_index], T.iloc[test_index]
                Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

                for model_y in my_list:
                    count = 0
                    for model_t in mt_list:
                        try:

                            if os.path.exists(results_file):
                                exists = results_df[
                                    (results_df["model_y"] ==
                                        model_y.__class__.__name__)
                                    & (results_df["model_t"] == model_t.__class__.__name__)
                                    & (results_df['fold'] == fold)
                                ].any().any()
                                if exists:
                                    print(
                                        f"Skipping model_y: {model_y}, model_t: {model_t}")
                                    continue
                            params_model_y = select_classification_hyperparameters(
                                model_y) if is_discrete else select_regression_hyperparameters(model_y)
                            params_model_t = select_classification_hyperparameters(
                                model_t)
                            grid_search_t = GridSearchCV(
                                model_t, params_model_t, cv=3)
                            grid_search_t.fit(X_train, Y_train)
                            best_params_t = grid_search_t.best_params_

                            grid_search_y = GridSearchCV(
                                model_y, params_model_y, cv=3)
                            grid_search_y.fit(X_train, Y_train)
                            best_params_y = grid_search_y.best_params_

                            model_t.set_params(**best_params_t)
                            model_y.set_params(**best_params_y)

                            temp_results = {'model_t': model_t.__class__.__name__,
                                            'model_y': model_y.__class__.__name__, 'fold': fold}

                            start = time.time()
                            model_t.fit(X_train, T_train)
                            end = time.time()
                            temp_results['time_model_t'] = end - start

                            start = time.time()
                            model_y.fit(X_train, Y_train)
                            end = time.time()
                            temp_results['time_model_y'] = end - start

                            T_pred = model_t.predict(X_train)
                            temp_results['f1_score_model_t'] = f1_score(
                                T_train, T_pred, average='weighted')  # or 'binary' for binary classification

                            Y_pred = model_y.predict(X_train)

                            if is_discrete:
                                temp_results['f1_score_model_y'] = f1_score(
                                    Y_train, Y_pred, average='weighted')  # or 'binary' for binary classification
                            else:
                                temp_results['mse_model_y'] = mean_squared_error(
                                    Y_train, Y_pred)

                            temp_results['best_params_t'] = best_params_t
                            temp_results['best_params_y'] = best_params_y

                            all_results.append(temp_results)
                            results_df = pd.DataFrame(all_results)
                            results_df.to_csv(
                                f'results/{key}_during_crossfit_params.csv')
                            print(
                                f"Completed running model_y: {model_y}, model_t: {model_t}")
                        except Exception as e:
                            print(
                                f"Error occurred while running {model_y}-{model_t} estimator: {str(e)}")
                        i += 1
                        count += 1
            results_df.to_csv(f"results/{key}_during_crossfit_params.csv")
            # wandb.alert(title="Code is done!", text="Code is done!")
        # wandb.finish()
== == == =


def select_classification_hyperparameters(estimator):
    """
    Returns a hyperparameter grid for the specified classification model type.

    Args:
        model_type (str): The type of model to be used. Valid values are 'linear', 'forest', 'nnet', and 'poly'.

    Returns:
        A dictionary representing the hyperparameter grid to search over.
    """
    if isinstance(estimator, LogisticRegressionCV) or estimator == 'linear':
        # Hyperparameter grid for linear classification model
        return {
            'Cs': [1, 10],
            'max_iter': [25]
        }
    elif isinstance(estimator, RandomForestClassifier) or estimator == 'forest':
        # Hyperparameter grid for random forest classification model
        return {
            'n_estimators': [25],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif isinstance(estimator, GradientBoostingClassifier) or estimator == 'gbf':
        # Hyperparameter grid for gradient boosting classification model
        return {
            'n_estimators': [100, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],

        }
    elif isinstance(estimator, MLPClassifier) or estimator == 'nnet':
        # Hyperparameter grid for neural network classification model
        return {
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [25]
        }
    else:
        warnings.warn("No hyperparameters for this type of model. There are default hyperparameters for LogisticRegressionCV, RandomForestClassifier, MLPClassifier, and the polynomial pipleine", category=UserWarning)
        return {}
        # raise ValueError("Invalid model type. Valid values are 'linear', 'forest', 'nnet', and 'poly'.")


def find_best_model_hyperparam(X, Y, model, is_discrete):
    if type(model) == str:
        if is_discrete:
            model = utils.select_discrete_estimator(model)
        else:
            model = utils.select_continuous_estimator(model)
    if is_discrete:
        hyperparam_grid_model = select_classification_hyperparameters(model)
    else:
        hyperparam_grid_model = utils.select_regression_hyperparameters(model)
    # print(model)
    grid_search = GridSearchCV(model, hyperparam_grid_model, cv=5, n_jobs=-1)
    grid_search.fit(X, Y.ravel())
    best_params = grid_search.best_params_
    current_best_model = grid_search.best_estimator_
    current_best_score = grid_search.best_score_

    # print(current_best_model, type(current_best_model))
    return current_best_model

# Divide into k folds.
# For each fold, select a different model and find the best hyperparameter
# get score, mse, time of estimators. Set cv=0 for estimator


def get_models_during_k_folds(X, T, Y, k, ci_estimator_list, model_y, model_t, model_y_discrete, model_t_discrete):
    cv = KFold(n_splits=k, shuffle=True, random_state=123)

    fold_models = {}
    i = 0
    total_start_time = time.time()
    for train_index, test_index in cv.split(X):
        X_train, T_train, Y_train = X.iloc[train_index], T.iloc[train_index], Y.iloc[train_index]
        X_test, T_test, Y_test = X.iloc[test_index], T.iloc[test_index], Y.iloc[test_index]
        causal_model_score = {}
        causal_model_mse = {}
        causal_model_time = {}

        if model_y_discrete:
            current_model_y = utils.select_discrete_estimator(model_y[i])
        else:
            current_model_y = utils.select_continuous_estimator(model_y[i])
        if model_t_discrete:
            current_model_t = utils.select_discrete_estimator(model_t[i])
        else:
            current_model_t = utils.select_continuous_estimator(model_t[i])

        # find the best hyperparameters for the first stage linear models
        best_model_y = find_best_model_hyperparam(
            X, Y, current_model_y, model_y_discrete)
        best_model_t = find_best_model_hyperparam(
            X, T, current_model_t, model_t_discrete)

        for ci in ci_estimator_list:
            causal_model = utils.get_estimators(ci, best_model_y, best_model_t)
            start_time = time.time()
            causal_model.fit(Y_train, T_train, X=X_train)
            run_time = time.time() - start_time
            te_pred = causal_model.effect(X_test)
            causal_model_mse[ci] = np.mean((Y_test - te_pred)**2)
            causal_model_time[ci] = run_time

        fold_models[f'fold {i}'] = {'model_y': best_model_y, 'model_t': best_model_t,
                                    'Mse': causal_model_mse, 'Runtime': causal_model_time}
        i += 1

    total_run_time = time.time() - total_start_time
    return fold_models, total_run_time

# Find best fold in terms of MSE and Runtime for each estimator


def find_best_fold_mse_runtime(fold_models, ci_estimator_list):
    mse_all_estimators = {}
    runtime_all_estimators = {}
    for ci in ci_estimator_list:
        mse_all_estimators[f'{ci}'] = []
        runtime_all_estimators[f'{ci}'] = []
    for k, value in fold_models.items():
        for ci in ci_estimator_list:
            mse_all_estimators[f'{ci}'].append(value['Mse'][ci])
            runtime_all_estimators[f'{ci}'].append(value['Runtime'][ci])

    best_mse_fold = {}
    best_runtime_fold = {}
    for ci in ci_estimator_list:
        best_mse_fold[ci] = np.argmin(mse_all_estimators[f'{ci}'])
        best_runtime_fold[ci] = np.argmin(runtime_all_estimators[f'{ci}'])

    best_models_mse = {}
    best_models_time = {}
    for ci in ci_estimator_list:
        fold_mse = best_mse_fold[ci]
        fold_time = best_runtime_fold[ci]

        if ci == 'sl' or ci == 'tl' or ci == 'xl':
            best_fold_model_t_mse = ''
            best_fold_model_t_time = ''
        else:
            best_fold_model_t_mse = fold_models[f'fold {fold_mse}']['model_t']
            best_fold_model_t_time = fold_models[f'fold {fold_time}']['model_t']

        best_models_mse[ci] = {'best_model_y': fold_models[f'fold {fold_mse}']['model_y'],
                               'best_model_t': best_fold_model_t_mse, 'Mse': fold_models[f'fold {fold_mse}']['Mse'][ci]}
        best_models_time[ci] = {'best_model_y': fold_models[f'fold {fold_time}']['model_y'],
                                'best_model_t': best_fold_model_t_time, 'Runtime': fold_models[f'fold {fold_time}']['Runtime'][ci]}

    return mse_all_estimators, best_mse_fold, best_models_mse, runtime_all_estimators, best_runtime_fold, best_models_time


def main(ci_estimator_list, model_y, model_t, key, k):

    data_dict = {'ihdp': utils.load_ihdp(), 'twin': utils.load_twin()}
    data, X, T, Y, true_ITE, true_ATE, true_ATE_stderr, is_discrete = data_dict[key]

    fold_models, total_run_time = get_models_during_k_folds(
        X, T, Y, k, ci_estimator_list, model_y, model_t, model_y_discrete=False, model_t_discrete=True)
    print(fold_models, total_run_time)

    mse_all_estimators, best_mse_fold, best_models_mse, runtime_all_estimators, best_runtime_fold, best_models_time = find_best_fold_mse_runtime(
        fold_models, ci_estimator_list)
    print(mse_all_estimators, best_mse_fold, best_models_mse)
    print(runtime_all_estimators, best_runtime_fold, best_models_time)

    results_file = f'../results/{key}_during_crossfit_params.csv'

    # Write MSE and runtime data to CSV file
    with open(results_file, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Estimator', 'Fold', 'model_y', 'model_t', 'MSE', 'Runtime', 'Best first-stage models Mse',
                           'Best first-stage models Runtime', 'Best Mse', 'Best Runtime', 'Total Runtime'])

        for estimator in mse_all_estimators:
            for i in range(k):
                mse = mse_all_estimators[estimator][i]
                runtime = runtime_all_estimators[estimator][i]
                csvwriter.writerow(
                    [estimator, i, fold_models[f'fold {i}']['model_y'], fold_models[f'fold {i}']['model_t'], mse, runtime, '', '', '', ''])
                i += 1

            best_mse = best_models_mse[estimator]['Mse']
            best_runtime = best_models_time[estimator]['Runtime']

            best_mse_models = f"{best_models_mse[estimator]['best_model_y']}\n\n{best_models_mse[estimator]['best_model_t']}"
            best_runtime_models = f"{best_models_time[estimator]['best_model_y']}\n\n{best_models_time[estimator]['best_model_t']}"

            csvwriter.writerow([estimator, f'Best Mse, Runtime Fold - {best_mse_fold[estimator]}, {best_runtime_fold[estimator]}',
                               '', '', '', '', best_mse_models, best_runtime_models, best_mse, best_runtime])

        csvwriter.writerow(['', '', '', '', '', '', '',
                           '', '', '', total_run_time])


if __name__ == "__main__":
    main(ci_estimator_list, model_y, model_t, key, k)

>>>>>> > 05b5aac2fdcc08ba579c173fe7ff044f9a0971e0
