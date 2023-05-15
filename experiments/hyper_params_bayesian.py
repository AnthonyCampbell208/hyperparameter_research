import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
from sklearn.linear_model import LinearRegression
from utils import *


def get_bayesian_hyperparams_discrete(X, Y, is_nnet):

    def lr_cv_score_logistic_regression(max_iter):
        model = LogisticRegressionCV(max_iter=int(max_iter))
        scores = cross_val_score(model, X, Y, cv=5, scoring='r2')
        mse_scores = -cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error', error_score='raise')
        return np.mean(mse_scores)

    def lr_cv_score_forest_classifier(n_estimators, max_depth, min_samples_split, min_samples_leaf):
        model = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth), min_samples_split=int(min_samples_split), min_samples_leaf=int(min_samples_leaf))
        scores = cross_val_score(model, X, Y, cv=5, scoring='r2')
        mse_scores = -cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')
        return np.mean(mse_scores)

    def lr_cv_score_gbf_classifier(n_estimators, learning_rate, max_depth):
        model = GradientBoostingClassifier(n_estimators=int(n_estimators), learning_rate=float(learning_rate), max_depth=int(max_depth))
        scores = cross_val_score(model, X, Y, cv=5, scoring='r2')
        mse_scores = -cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')
        return np.mean(mse_scores)

    def lr_cv_score_nnet_classifier(hidden_layer_sizes, alpha, max_iter):
        model = MLPClassifier(hidden_layer_sizes=int(hidden_layer_sizes), alpha=float(alpha), max_iter=int(max_iter))
        scores = cross_val_score(model, X, Y, cv=5, scoring='r2')
        mse_scores = -cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')
        return np.mean(mse_scores)

    #define the parameter bounds for optimization
    pbounds_logistic = {
        'Cs': (0.001, 10),
        # 'penalty': 'elasticnet',
        # 'solver': ['lbfgs', 'liblinear', 'saga'],
        'max_iter': (25, 100)
    }

    pbounds_forest_class = {
        'n_estimators': (25, 100),
        'max_depth': (5, 20),
        'min_samples_split': (2, 5),
        'min_samples_leaf': (1, 2)
    }

    pbounds_gbf_class = {
        'n_estimators': (100, 500),
        'learning_rate': (0.01, 0.1),
        'max_depth': (3, 7),
    }

    pbounds_nnet_class = {
        'hidden_layer_sizes': (10, 100),
        'alpha': (0.0001, 0.01),
        'max_iter': (5, 100)
    }

    # Create the Bayesian optimization objects
    # optimizer = BayesianOptimization(
    #     f=lr_cv_score_logistic_regression,
    #     pbounds=pbounds_logistic,
    #     random_state=42,
    # )
    # optimizer.maximize(init_points=5, n_iter=20)
    # best_params = optimizer.max['params']
    # # best_Cs = best_params['Cs']
    # best_max_iter = best_params['max_iter']
    # print("Best Hyperparameters for Logistic Regression:")
    # # print(f"Cs: {best_Cs}")
    # print(f"max_iter: {best_max_iter}")

    optimizer1 = BayesianOptimization(
    f=lr_cv_score_forest_classifier,
    pbounds=pbounds_forest_class,
    random_state=42,
    )
    optimizer1.maximize(init_points=5, n_iter=10)
    best_params1 = optimizer1.max['params']
    best_n_estimators = best_params1['n_estimators']
    best_max_depth = best_params1['max_depth']
    print("Best Hyperparameters for Random Forest Classifier:")
    print(f"n_estimators: {best_n_estimators}")
    print(f"max_depth: {best_max_depth}")

    optimizer2 = BayesianOptimization(
        f=lr_cv_score_gbf_classifier,
        pbounds=pbounds_gbf_class,
        random_state=42,
    )
    optimizer2.maximize(init_points=5, n_iter=10)
    best_params2 = optimizer2.max['params']
    best_n_estimators = best_params2['n_estimators']
    best_learning_rate = best_params2['learning_rate']
    best_max_depth = best_params2['max_depth']
    print("Best Hyperparameters for Gradient Boosting Classifier:")
    print(f"n_estimators: {best_n_estimators}")
    print(f"learning_rate: {best_learning_rate}")
    print(f"max_depth: {best_max_depth}")

    if is_nnet:
        optimizer3 = BayesianOptimization(
        f=lr_cv_score_nnet_classifier,
        pbounds=pbounds_nnet_class,
        random_state=42,
        )
        optimizer3.maximize(init_points=5, n_iter=10)
        best_params3 = optimizer3.max['params']
        best_hidden_layer_sizes = best_params3['hidden_layer_sizes']
        best_alpha = best_params3['alpha']
        best_max_iter = best_params3['max_iter']
        print("Best Hyperparameters for Neural Net classifier:")
        print(f"hidden_layer_sizes: {best_hidden_layer_sizes}")
        print(f"alpha: {best_alpha}")
        print(f"max_iter: {best_max_iter}")
    else:
        best_params3 = []

    return best_params1, best_params2, best_params3


def get_bayesian_hyperparams_continuous(X, Y, is_nnet):

    def lr_cv_score_elastic(l1_ratio, max_iter):
        model = ElasticNetCV(l1_ratio=l1_ratio, max_iter=int(max_iter))
        scores = cross_val_score(model, X, Y, cv=5, scoring='r2')
        mse_scores = -cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error', error_score='raise')
        return np.mean(mse_scores)

    def lr_cv_score_forest_regressor(n_estimators, max_depth, min_samples_split):
        model = RandomForestRegressor(n_estimators=int(n_estimators), max_depth=int(max_depth), min_samples_split=int(min_samples_split))
        scores = cross_val_score(model, X, Y, cv=5, scoring='r2')
        mse_scores = -cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')
        return np.mean(mse_scores)

    def lr_cv_score_gbf_regressor(n_estimators, learning_rate, max_depth):
        model = GradientBoostingRegressor(n_estimators=int(n_estimators), learning_rate=float(learning_rate), max_depth=int(max_depth))
        scores = cross_val_score(model, X, Y, cv=5, scoring='r2')
        mse_scores = -cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')
        return np.mean(mse_scores)

    def lr_cv_score_nnet_regressor(hidden_layer_sizes, alpha, max_iter):
        model = MLPRegressor(hidden_layer_sizes=int(hidden_layer_sizes), alpha=float(alpha), max_iter=int(max_iter))
        scores = cross_val_score(model, X, Y, cv=5, scoring='r2')
        mse_scores = -cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')
        return np.mean(mse_scores)

    #define the parameter bounds for optimization
    pbounds_elastic = {
        'l1_ratio': (0.1, 0.9),
        'max_iter': (10, 50),
    }

    pbounds_forest_reg = {
        'n_estimators': (5, 25),
        'max_depth': (0, 50),
        'min_samples_split': (2, 10)
    }

    pbounds_gbf_reg = {
        'n_estimators': (10, 100),
        'learning_rate': (0.01, 1.0),
        'max_depth': (3, 5),
    }

    pbounds_nnet_reg = {
        'hidden_layer_sizes': (10, 100),
        'alpha': (0.0001, 0.01),
        # 'learning_rate': ['constant', 'adaptive'],
        'max_iter': (5, 25)
    }

    #Create the Bayesian optimization objects
    optimizer = BayesianOptimization(
        f=lr_cv_score_elastic,
        pbounds=pbounds_elastic,
        random_state=42,
    )
    optimizer.maximize(init_points=5, n_iter=20)
    best_params = optimizer.max['params']
    best_l1_ratio = best_params['l1_ratio']
    best_max_iter = best_params['max_iter']
    print("Best Hyperparameters for Logistic Regression:")
    print(f"l1_ratio: {best_l1_ratio}")
    print(f"max_iter: {best_max_iter}")

    optimizer1 = BayesianOptimization(
    f=lr_cv_score_forest_regressor,
    pbounds=pbounds_forest_reg,
    random_state=42,
    )
    optimizer1.maximize(init_points=5, n_iter=10)
    best_params1 = optimizer1.max['params']
    best_n_estimators = best_params1['n_estimators']
    best_max_depth = best_params1['max_depth']
    print("Best Hyperparameters for Random Forest Classifier:")
    print(f"n_estimators: {best_n_estimators}")
    print(f"max_depth: {best_max_depth}")

    optimizer2 = BayesianOptimization(
        f=lr_cv_score_gbf_regressor,
        pbounds=pbounds_gbf_reg,
        random_state=42,
    )
    optimizer2.maximize(init_points=5, n_iter=10)
    best_params2 = optimizer2.max['params']
    best_n_estimators = best_params2['n_estimators']
    best_learning_rate = best_params2['learning_rate']
    best_max_depth = best_params2['max_depth']
    print("Best Hyperparameters for Gradient Boosting Classifier:")
    print(f"n_estimators: {best_n_estimators}")
    print(f"learning_rate: {best_learning_rate}")
    print(f"max_depth: {best_max_depth}")

    if is_nnet:
        optimizer3 = BayesianOptimization(
        f=lr_cv_score_nnet_regressor,
        pbounds=pbounds_nnet_reg,
        random_state=42,
        )
        optimizer3.maximize(init_points=5, n_iter=10)
        best_params3 = optimizer3.max['params']
        best_hidden_layer_sizes = best_params3['hidden_layer_sizes']
        best_alpha = best_params3['alpha']
        best_max_iter = best_params3['max_iter']
        print("Best Hyperparameters for Neural Net classifier:")
        print(f"hidden_layer_sizes: {best_hidden_layer_sizes}")
        print(f"alpha: {best_alpha}")
        print(f"max_iter: {best_max_iter}")
    else:
        best_params3 = []

    return best_params, best_params1, best_params2, best_params3