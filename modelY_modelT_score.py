import time
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def causal_inference_experiment(model_y, model_t, x, y, t, param_y, param_t):
    model_y.set_params(**param_y)
    model_t.set_params(**param_t)

    start_time_y = time.time()
    model_y.fit(x, y)
    runtime_y = time.time() - start_time_y

    start_time_t = time.time()
    model_t.fit(x, t)
    runtime_t = time.time() - start_time_t

    y_pred = model_y.predict(x)
    t_pred = model_t.predict(x)

    mse_y = mean_squared_error(y, y_pred)
    rmse_y = np.sqrt(mse_y)
    r2_y = r2_score(y, y_pred)

    mse_t = mean_squared_error(t, t_pred)
    rmse_t = np.sqrt(mse_t)
    r2_t = r2_score(t, t_pred)

    return {
        "runtime_y": runtime_y,
        "mse_y": mse_y,
        "rmse_y": rmse_y,
        "r2_y": r2_y,
        "runtime_t": runtime_t,
        "mse_t": mse_t,
        "rmse_t": rmse_t,
        "r2_t": r2_t,
    }


if __name__ == "__main__":
    models_and_parameters = get_models_and_parameters()
    X = np.random.rand(100, 10)
    Y = np.random.rand(100)

    best_model = find_best_model(models_and_parameters, X, Y)
    print(best_model)