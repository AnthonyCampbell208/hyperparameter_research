import time
from utils import *
import pdb
import sklearn
from sklearn.preprocessing import StandardScaler
# from data_preprocessing.ihdp import * 
# from data_preprocessing.lalonde import *
# from data_preprocessing.lbidd import *
# from data_preprocessing.synthetic import *
# from data_preprocessing.twins import *

k = 2
ci_estimators = ['sl', 'tl', 'xl', 'dml', 'sparse_dml', 'kernel_dml', 'CausalForestDML']

# ci_estimators = ['dr']

def causal_inference_analysis(model_y, model_t, str_causal_model,x, y, t, true_ate, true_ate_std, true_ite, is_meta):
    # set hyperparameters
    causal_model = get_estimators(str_causal_model, model_y, model_t)
    if is_meta:
        start_time = time.time()
        causal_model.fit(y, t, X=x)
        run_time = time.time() - start_time
    else:
        start_time = time.time()
        causal_model.fit(y, t, X=x, W=None)
        run_time = time.time() - start_time

    estimated_ate = causal_model.ate(x)

    estimated_ite_values = causal_model.effect(x)

    tao_risk, mu_risk = calculate_risks(true_ate, estimated_ate, true_ite, estimated_ite_values)

    return {'causal_model_name': causal_model.__class__.__name__, 'model_t': model_t.__class__.__name__, 'model_y': model_y.__class__.__name__,
            'est_ate': estimated_ate, 'true_ate': true_ate, 'mu_risk': mu_risk, 'tao_risk': tao_risk, 
            'run_time': run_time}, estimated_ite_values

    

if __name__ == "__main__":
    classifiers = [GradientBoostingClassifier(),
               RandomForestClassifier(),
               LogisticRegressionCV(),
               MLPClassifier()]

    regressors = [GradientBoostingRegressor(),
                RandomForestRegressor(),
                LinearRegression(),
                ElasticNet(),
                ElasticNetCV(),
                MLPRegressor()]


    # data_dict = {'ihdp':load_ihdp()}
    data_dict = {'twin':load_twin()}
    all_results = []
    for key in data_dict:
        data, X, T, Y, true_ite, true_ATE, true_ATE_stderr, is_discrete = data_dict[key]
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(X)

        my_list = regressors
        mt_list = classifiers if is_discrete else regressors
        for str_causal_model in ci_estimators:
            is_meta = False
            if str_causal_model in ['sl', 'xl', 'tl']:
                    is_meta = True
            for model_y  in my_list:
                count = 0
                for model_t in mt_list:
                    """
                    Set hyperparameters for model_y and model_t 

                    model_y is regressor so get reg hyperparams

                    model_t could be either (contingent on is_dsicrete)

                    Method "set_params" simply sets the parameters for the model (native sk_learn)

                    """
                    model_y.set_params(**select_regression_hyperparameters(model_y))
                    if is_discrete:
                        model_t.set_params(**select_classification_hyperparameters(model_t))
                    else:
                        model_t.set_params(**select_regression_hyperparameters(model_t))
                        
                    try:
                        temp_results, estimated_ite_values = causal_inference_analysis(model_y, model_t, str_causal_model, x_scaled, Y, T, true_ATE, true_ATE_stderr, true_ite, is_meta)
                        temp_results['data'] = key
                        all_results.append(temp_results)
                        results_df = pd.DataFrame(all_results)
                        results_df.to_csv(f'results/{key}_no_params_baselines.csv')
                        print(f"Completed running model_y: {model_y}, model_t: {model_t}, str_causal_model: {str_causal_model}")
                    except Exception as e:
                        print(f"Error occurred while running {model_y}-{model_t} estimator with {str_causal_model} method: {str(e)}")
        
            



