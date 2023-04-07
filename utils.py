import os
import logging
import argparse
import numpy as np
import pandas as pd
import os
os.chdir('/Users/anthonycampbell/Downloads')

import econml
from econml.dml import LinearDML, SparseLinearDML, KernelDML, CausalForestDML
from econml.orf import DMLOrthoForest
from econml.dr import DRLearner
from econml.metalearners import SLearner, TLearner, XLearner
from econml.grf import CausalForest
from sklearn.model_selection import KFold
# from models.data import IHDP, JOBS, TWINS, NEWS
# from models.estimators import SEvaluator, TEvaluator, XEvaluator, DREvaluator, DMLEvaluator, IPSWEvaluator, CausalForestEvaluator
# from models.estimators import TSEvaluator, DRSEvaluator, DMLSEvaluator, IPSWSEvaluator
# from helpers.utils import init_logger, get_model_name

import pandas as pd
from sklearn.model_selection import train_test_split

def load_401k():
    url = "https://raw.githubusercontent.com/CausalAIBook/MetricsMLNotebooks/main/data/401k.csv"
    data_df = pd.read_csv(url)
    
    feature_cols = ["age", "inc", "fsize", "educ", "db", "marr", "male", "twoearn", "pira", "nohs", "hs", "smcol", "col", "hown"]
    X = data_df[feature_cols]
    T = data_df['p401']
    Y = data_df['net_tfa']

    return X, T, Y

def load_abalone():
    data_df = pd.read_csv("abalone.csv")
    col =  ["Sex", "Length", "Diameter", "Height", "Whole_weight","Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
    data_df.columns = col
    feature_col = ["Length", "Diameter", "Height"]
    outcome_col = ["Rings"]
    treatment_col = ["Whole_weight"]
    X = data_df[feature_col]
    Y = data_df[outcome_col].values.reshape(-1, 1)
    T = data_df[treatment_col].values.reshape(-1, 1)

    return data_df, X, T, Y


def load_ihdp():
    url = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv"
    ihdp_df = pd.read_csv(url)
    col =  ["treatment", "y_factual", "y_cfactual", "mu0", "mu1","x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25"]
    ihdp_df.columns = col
    ihdp_df = ihdp_df.astype({"treatment":'bool'}, copy=False)
    T = ihdp_df['treatment']
    feature_cols = ["mu0", "mu1", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23", "x24", "x25"]
    X = ihdp_df[feature_cols]

    #Column y_factual gives the observed Bayley Mental Development Index score
    Y = ihdp_df['y_factual']

    return ihdp_df, X, T, Y


def get_estimators(estimation_model, model_y, model_t):
    if estimation_model == 'sl':
        return SLearner(overall_model = model_y)
    elif estimation_model == 'tl':
        return TLearner(models= model_y)
    elif estimation_model == 'xl':
        return XLearner(models= model_y)
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