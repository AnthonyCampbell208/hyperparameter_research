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
