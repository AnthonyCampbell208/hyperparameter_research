import time
from utils import *

import time

k = 2
ci_estimators = ['sl', 'tl', 'xl', 'dml', 'orf', 'dr', 'sparse_dml', 'kernel_dml', 'CausalForestDML']

def causal_inference_analysis(model_y, model_t, str_causal_model,x, w, y, t):
    causal_model = get_estimators(str_causal_model, model_y, model_t)

    start_time = time.time()
    causal_model.fit(x, t, y, W=w)
    run_time = time.time() - start_time

    ate = causal_model.ate_
    std_ate = causal_model.ate_stderr_
    
    true_ate = ...
    true_std_ate = ...

    return {'combo': f"{model_t.__class__.__name__}_{model_y.__class__.__name__}",
        'ate': ate, 'mu_risk': mu_risk, 'tao_risk': tao_risk, 'run_time': run_time, 
        'model_t': model_t.__class__.__name__, 'model_y': model_y.__class__.__name__}


if __name__ == "__main__":
    kf = KFold(n_splits=k)
    my_list = []
    mt_list = []
    data_list = []
    all_results = []
    for tuple_data in data_list:
        data, x, w, y, t = tuple_data
        for str_causal_model in ci_estimators:
            is_meta = False
            if str_causal_model in ['sl', 'xl', 'tl']:
                 is_meta = True
            for model_y  in my_list:
                count = 0
                for model_t in mt_list:
                    # if is_meta:
                    #     count += 1
                    #     if count > 1:
                    #         continue
                    # # for idx, (train_index, test_index) in enumerate(kf.split(data)):
                    # #     train_data = data.iloc[train_index]
                    #     # test_data = data.iloc[test_index]
                    try:
                        temp_results = causal_inference_analysis(model_y, model_t, str_causal_model, x, w, y, t)
                        all_results.append(temp_results)
                    except Exception as e:
                        print(f"Error occurred while running {model_y}-{model_t} estimator with {str_causal_model} method: {str(e)}")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('results/no_params_baselines.csv')



