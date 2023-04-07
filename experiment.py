import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


def get_models_and_parameters():
    alpha = [0.001, 0.01, 0.1, 0.5, 1, 2, 10, 20]

    model_dict = {'elastic': {'model': ElasticNetCV(), 'parameters': {}},
        'forest': {'model': RandomForestRegressor(),
        'parameters': {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
        'min_samples_leaf': [0.01,0.02,0.03,0.04, 0.05,1,2,3, 4,5,6,7,8,9]}},
        'gbf': {'model': GradientBoostingRegressor(),
        'parameters': {'n_estimators': [100, 300, 500, 700, 1000],
        'max_depth': [2, 4, 6, 8, 10],
        'min_samples_split': [5, 10, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 1]}},
        'nnet': {'model': MLPRegressor(),
        'parameters': {'hidden_layer_sizes': [4, 8, 16, 32, 64, 128],
        'learning_rate_init': [0.0001, 0.001],
        'batch_size': [32, 64, 128, 250]}}}

    return models_and_parameters


def find_best_model(models_and_parameters, X, Y):
    best_model = None
    best_mse = None

    for name, model, param_grid in models_and_parameters:
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X, Y)
        current_best_model = grid_search.best_estimator_
        predictions = current_best_model.predict(X)
        current_best_mse = mean_squared_error(Y, predictions)

        if best_mse is None or current_best_mse < best_mse:
            best_model = current_best_model
            best_mse = current_best_mse

    return best_model



if __name__ == "__main__":
    models_and_parameters = get_models_and_parameters()
    X = np.random.rand(100, 10)
    Y = np.random.rand(100)

    best_model = find_best_model(models_and_parameters, X, Y)
    print(best_model)
