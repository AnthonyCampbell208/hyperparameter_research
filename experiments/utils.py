from typing import Iterable, Any
from itertools import product
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler,
                                   PolynomialFeatures, RobustScaler,
                                   StandardScaler)
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import (BaseCrossValidator, GridSearchCV, KFold,
                                     RandomizedSearchCV, StratifiedKFold,
                                     check_cv, train_test_split)
from sklearn.linear_model import (ARDRegression, BayesianRidge, ElasticNet,
                                  ElasticNetCV, Lars, Lasso, LassoLars,
                                  LinearRegression, LogisticRegression,
                                  LogisticRegressionCV,
                                  OrthogonalMatchingPursuit, Ridge)
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.base import BaseEstimator, is_regressor
from econml.orf import DMLOrthoForest
from econml.metalearners import SLearner, TLearner, XLearner
from econml.grf import CausalForest
from econml.dr import DRLearner
from econml.dml import CausalForestDML, KernelDML, LinearDML, SparseLinearDML
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.linear_model
import sklearn.ensemble
import sklearn
import econml
import argparse
import logging
import os
import pdb
import numpy as np
import pandas as pd
import pickle
# os.chdir('/Users/anthonycampbell/Downloads')

import warnings

import sys
sys.path.insert(
    0, '/Users/anthonycampbell/miniforge3/pkgs/econml-0.13.1-py39h533cade_0/lib/python3.9/site-packages/')


# from models.data import IHDP, JOBS, TWINS, NEWS
# from models.estimators import SEvaluator, TEvaluator, XEvaluator, DREvaluator, DMLEvaluator, IPSWEvaluator, CausalForestEvaluator
# from models.estimators import TSEvaluator, DRSEvaluator, DMLSEvaluator, IPSWSEvaluator
# from helpers.utils import init_logger, get_model_name


def load_ihdp():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header=None)
    col = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1", ]
    for i in range(1, 26):
        col.append("x"+str(i))
    data.columns = col
    data = data.astype({"treatment": 'bool'}, copy=False)
    data.head()

    treatment = 'treatment'
    outcome = 'y_factual'

    common_causes = ["x"+str(i) for i in range(1, 26)]

    X = data[common_causes]
    Y = data[outcome]
    T = data[treatment].astype(int)

    true_ite = data['mu1'] - data['mu0']
    true_ATE = np.mean(true_ite)
    true_ATE_stderr = np.std(true_ite)

    is_discrete = True

    return data, X, T, Y, true_ite, true_ATE, true_ATE_stderr, is_discrete


def load_twin():
    # The covariates data has 46 features
    x = pd.read_csv(
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_X_3years_samesex.csv")

    # The outcome data contains mortality of the lighter and heavier twin
    y = pd.read_csv(
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_Y_3years_samesex.csv")

    # The treatment data contains weight in grams of both the twins
    t = pd.read_csv(
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_T_3years_samesex.csv")

    lighter_columns = ['pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
                       'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
                       'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
                       'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
                       'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
                       'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
                       'data_year', 'nprevistq', 'dfageq', 'feduc6', 'infant_id_0',
                       'dlivord_min', 'dtotord_min', 'bord_0',
                       'brstate_reg', 'stoccfipb_reg', 'mplbir_reg']
    heavier_columns = ['pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
                       'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
                       'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
                       'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
                       'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
                       'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
                       'data_year', 'nprevistq', 'dfageq', 'feduc6',
                       'infant_id_1', 'dlivord_min', 'dtotord_min', 'bord_1',
                       'brstate_reg', 'stoccfipb_reg', 'mplbir_reg']

    # Since data has pair property,processing the data to get separate row for each twin so that each child can be treated as an instance
    data = []

    for i in range(len(t.values)):

        # select only if both <=2kg
        if t.iloc[i].values[1] >= 2000 or t.iloc[i].values[2] >= 2000:
            continue

        this_instance_lighter = list(x.iloc[i][lighter_columns].values)
        this_instance_heavier = list(x.iloc[i][heavier_columns].values)

        # adding weight
        this_instance_lighter.append(t.iloc[i].values[1])
        this_instance_heavier.append(t.iloc[i].values[2])

        # adding treatment, is_heavier
        this_instance_lighter.append(0)
        this_instance_heavier.append(1)

        # adding the outcome
        this_instance_lighter.append(y.iloc[i].values[1])
        this_instance_heavier.append(y.iloc[i].values[2])
        data.append(this_instance_lighter)
        data.append(this_instance_heavier)
    cols = ['pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
            'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
            'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
            'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
            'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
            'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
            'data_year', 'nprevistq', 'dfageq', 'feduc6',
            'infant_id', 'dlivord_min', 'dtotord_min', 'bord',
            'brstate_reg', 'stoccfipb_reg', 'mplbir_reg', 'wt', 'treatment', 'outcome']
    df = pd.DataFrame(columns=cols, data=data)
    # explicitly assigning treatment column as boolean
    df = df.astype({"treatment": 'bool'}, copy=False)

    df.fillna(value=df.mean(), inplace=True)  # filling the missing values
    df.fillna(value=df.mode().loc[0], inplace=True)

    data_1 = df[df["treatment"] == 1].reset_index()
    data_0 = df[df["treatment"] == 0].reset_index()

    true_ITE = data_1["outcome"] - data_0["outcome"]
    true_ATE = np.mean(true_ITE)
    true_ATE_stderr = np.std(true_ITE)

    T = df['treatment'].astype(int)
    Y = df['outcome']
    X = df.drop(['treatment', 'outcome'], axis=1)
    is_discrete = True
    return (data, X, T, Y, true_ITE, true_ATE, true_ATE_stderr, is_discrete)


def load_acic():
    acic_datalist = [
        'ACIC_dataset/high_binary_datasets.pickle',
        'ACIC_dataset/low_binary_datasets.pickle',
        'ACIC_dataset/high_continuous_datasets.pickle',
        'ACIC_dataset/low_continuous_datasets.pickle'
    ]

    highDim_trueATE = pd.read_csv("ACIC_dataset/true_ate/highDim_trueATE.csv")
    lowDim_trueATE = pd.read_csv("ACIC_dataset/true_ate/lowDim_trueATE.csv")

    for file in acic_datalist:
        with open(file, 'rb') as f:
            data_file_list = pickle.load(f)

        for j, data_file in enumerate(data_file_list):
            data = pd.read_csv(data_file)
            y_col = 'Y'
            treatment_col = 'A'
            covariate_cols = data.columns.drop([y_col, treatment_col])

            X = data[covariate_cols]
            Y = data[y_col]
            T = data[treatment_col].astype(int)

            file_name = os.path.basename(data_file).split('.')[0]
            if 'high' in file:
                true_ate = highDim_trueATE.loc[highDim_trueATE['filename']
                                               == file_name, 'trueATE'].values[0]
            else:
                true_ate = lowDim_trueATE.loc[lowDim_trueATE['filename']
                                              == file_name, 'trueATE'].values[0]

            # For binary datasets, is_discrete is True, for continuous ones it is False
            is_discrete = True if 'binary' in file else False

            yield data, X, T, Y, None, true_ate, None, is_discrete, file_name


def calculate_risks(true_ate, estimated_ate, true_ite_values, estimated_ite_values):
    """
    Calculates the tau risk and mu risk for given true and estimated ATE and ITE values.

    Args:
    true_ate (float): True ATE value.
    estimated_ate (float): Estimated ATE value.
    true_ite_values (numpy array): Array of true ITE values.
    estimated_ite_values (numpy array): Array of estimated ITE values.

    Returns:
    tau_risk (float): Calculated tau risk.
    mu_risk (float): Calculated mu risk.
    """

    # Compute tau risk
    tau_risk = (true_ate - estimated_ate) ** 2

    # Compute mu risk
    if true_ite_values == None:
        mu_risk = None
    else:
        if len(true_ite_values) != len(estimated_ite_values):
            mu_risk = None
        else:
            mu_risk = np.mean((true_ite_values - estimated_ite_values) ** 2)

    return tau_risk, mu_risk


def get_estimators(estimation_model, model_y, model_t):
    if estimation_model == 'sl':
        return SLearner(overall_model=model_y)
    elif estimation_model == 'tl':
        return TLearner(models=model_y)
    elif estimation_model == 'xl':
        return XLearner(models=model_y)
    elif estimation_model == 'dml':
        return LinearDML(model_y=model_y, model_t=model_t)
    elif estimation_model == 'orf':
        return DMLOrthoForest(model_y=model_y, model_t=model_t)
    elif estimation_model == 'dr':
        return DRLearner(model_y=model_y, model_t=model_t)
    elif estimation_model == 'sparse_dml':
        return SparseLinearDML(model_y=model_y, model_t=model_t)
    elif estimation_model == 'kernel_dml':
        return KernelDML(model_y=model_y, model_t=model_t)
    elif estimation_model == 'CausalForestDML':
        return CausalForestDML(model_y=model_y, model_t=model_t)
    else:
        raise ValueError("Unrecognized 'estimation_model' key.")


def select_continuous_estimator(estimator_type):
    """
    Returns a continuous estimator object for the specified estimator type.

    Args:
        estimator_type (str): The type of estimator to use, one of: 'linear', 'forest', 'gbf', 'nnet', 'poly'.

    Returns:
        object: An instance of the selected estimator class.

    Raises:
        ValueError: If the estimator type is unsupported.
    """
    if estimator_type == 'linear':
        return (ElasticNetCV())
    elif estimator_type == 'forest':
        return RandomForestRegressor()
    elif estimator_type == 'gbf':
        return GradientBoostingRegressor()
    elif estimator_type == 'nnet':
        return (MLPRegressor())
    elif estimator_type == 'poly':
        poly = sklearn.preprocessing.PolynomialFeatures()
        # Play around with precompute and tolerance
        linear = sklearn.linear_model.ElasticNetCV(cv=3)
        return (Pipeline([('poly', poly), ('linear', linear)]))
    else:
        raise ValueError(f"Unsupported estimator type: {estimator_type}")


def select_discrete_estimator(estimator_type):
    """
    Returns a discrete estimator object for the specified estimator type.

    Args:
        estimator_type (str): The type of estimator to use, one of: 'linear', 'forest', 'gbf', 'nnet', 'poly'.

    Returns:
        object: An instance of the selected estimator class.

    Raises:
        ValueError: If the estimator type is unsupported.
    """
    if estimator_type == 'linear':
        return (LogisticRegressionCV(multi_class='auto'))
    elif estimator_type == 'forest':
        return RandomForestClassifier()
    elif estimator_type == 'gbf':
        return GradientBoostingClassifier()
    elif estimator_type == 'nnet':
        return (MLPClassifier())
    elif estimator_type == 'poly':
        poly = PolynomialFeatures()
        linear = LogisticRegressionCV(multi_class='auto')
        return (Pipeline([('poly', poly), ('linear', linear)]))
    else:
        raise ValueError(f"Unsupported estimator type: {estimator_type}")


def select_classification_hyperparameters(estimator):
    """
    Returns a hyperparameter grid for the specified classification model type.

    Args:
        model_type (str): The type of model to be used. Valid values are 'linear', 'forest', 'nnet', and 'poly'.

    Returns:
        A dictionary representing the hyperparameter grid to search over.
    """

    if isinstance(estimator, LogisticRegressionCV):
        # Hyperparameter grid for linear classification model
        return {
            'Cs': [0.01, 0.1, 1],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['lbfgs', 'liblinear', 'saga']
        }
    elif isinstance(estimator, RandomForestClassifier):
        # Hyperparameter grid for random forest classification model
        return {
            'n_estimators': [100, 500],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif isinstance(estimator, GradientBoostingClassifier):
        # Hyperparameter grid for gradient boosting classification model
        return {
            'n_estimators': [100, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],

        }
    elif isinstance(estimator, MLPClassifier):
        # Hyperparameter grid for neural network classification model
        return {
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    elif isinstance(estimator, RandomForestClassifier):
        return { 'n_estimators': [10]}
    else:
        warnings.warn("No hyperparameters for this type of model. There are default hyperparameters for LogisticRegressionCV, RandomForestClassifier, MLPClassifier, and the polynomial pipleine", category=UserWarning)
        return {}
        # raise ValueError("Invalid model type. Valid values are 'linear', 'forest', 'nnet', and 'poly'.")


def select_regression_hyperparameters(estimator):
    """
    Returns a dictionary of hyperparameters to be searched over for a regression model.

    Args:
        model_type (str): The type of model to be used. Valid values are 'linear', 'forest', 'nnet', and 'poly'.

    Returns:
        A dictionary of hyperparameters to be searched over using a grid search.
    """
    if isinstance(estimator, ElasticNetCV):
        return {
            'l1_ratio': [0.1, 0.5, 0.9],
            'max_iter': [1000],
        }
    elif isinstance(estimator, RandomForestRegressor):
        return {
            'n_estimators': [50],
            'max_depth': [None, 10, 50],
            'min_samples_split': [2, 5, 10],
        }
    elif isinstance(estimator, MLPRegressor):
        # Hyperparameter grid for neural network classification model
        return {
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    elif isinstance(estimator, GradientBoostingRegressor):
        # Hyperparameter grid for gradient boosting regression model
        return {
            'n_estimators': [100],
            'learning_rate': [0.01, 0.1, 1.0],
            'max_depth': [3, 5],
        }
    elif isinstance(estimator, RandomForestRegressor):
        return { 'n_estimators': [10]}
    else:
        warnings.warn("No hyperparameters for this type of model. There are default hyperparameters for ElasticNetCV, RandomForestRegressor, MLPRegressor, and the polynomial pipeline.", category=UserWarning)
        return {}
        # raise ValueError("Invalid model type. Valid values are 'linear', 'forest', 'nnet', and 'poly'.")


# return all combos


# : dict[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
def grid_parameters(parameters):
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))
