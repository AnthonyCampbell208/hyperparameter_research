import time
from utils import *
import pdb
import os
from sklearn.preprocessing import StandardScaler
import sys
import wandb
# sys.path.insert(
#     0, '/Users/anthonycampbell/miniforge3/pkgs/econml-0.13.1-py39h533cade_0/lib/python3.9/site-packages/')

k = 2
ci_estimators = ['tl', 'dml', 'kernel_dml', 'CausalForestDML']

# ci_estimators = ['dr']


def causal_inference_analysis(model_y, model_t, causal_model, x, y, t, true_ate, true_ate_std, true_ite, is_meta):
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

    tao_risk, mu_risk = calculate_risks(
        true_ate, estimated_ate, true_ite, estimated_ite_values)

    if is_meta:
        return {'causal_model_name': causal_model.__class__.__name__, 'model_t': None, 'model_y': model_y.__class__.__name__,
                'est_ate': estimated_ate, 'true_ate': true_ate, 'mu_risk': mu_risk, 'tao_risk': tao_risk,
                'run_time': run_time}, estimated_ite_values
    else:
        return {'causal_model_name': causal_model.__class__.__name__, 'model_t': model_t.__class__.__name__, 'model_y': model_y.__class__.__name__,
                'est_ate': estimated_ate, 'true_ate': true_ate, 'mu_risk': mu_risk, 'tao_risk': tao_risk,
                'run_time': run_time}, estimated_ite_values


def combination_exists_in_results(key, model_y, model_t, str_causal_model):

    results_df = pd.read_csv(results_file)
    exists = results_df[
        (results_df["model_y"] == model_y.__class__.__name__)
        & (results_df["model_t"] == model_t.__class__.__name__)
        & (results_df["causal_model_name"] == str_causal_model)
    ].any().any()

    return exists


if __name__ == "__main__":

    classifiers = [RandomForestClassifier(),
                   LogisticRegressionCV(max_iter=25),
                   #    MLPClassifier(max_iter=25),
                   'auto']

    regressors = [RandomForestRegressor(),
                  ElasticNetCV(max_iter=100),
                  #   MLPRegressor(max_iter=25),
                  'auto']

    wandb.init(project="cs696ds-econml", config={
        "causal_estimators": ci_estimators,
        "classifiers": classifiers,
        "regressors": regressors,
    })
    config = wandb.config
    np.random.seed = 42
    # data_dict = {'ihdp':load_ihdp()}
    # data_dict = {'twin':load_twin()}
    data_dict = {'acic': None}

    for key in data_dict:
        if key == 'acic':
            iterator = load_acic()
            for acic_data in iterator:
                data, X, T, Y, true_ite, true_ATE, true_ATE_stderr, is_discrete, file_name = acic_data
                scaler = StandardScaler()
                x_scaled = scaler.fit_transform(X)
                results_file = f'results/{key}_no_params_baselines.csv'
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
                for str_causal_model in ci_estimators:
                    is_meta = False
                    if str_causal_model in ['sl', 'xl', 'tl']:
                        is_meta = True
                    for model_y in my_list:
                        count = 0
                        for model_t in mt_list:
                            if is_meta and count >= 1:
                                continue
                            try:
                                causal_model = get_estimators(
                                    str_causal_model, model_y, model_t)
                                exists = False
                                if os.path.exists(results_file):
                                    if is_meta:
                                        exists = results_df[
                                            (results_df["model_y"] == model_y.__class__.__name__) & (
                                                results_df["causal_model_name"] == causal_model.__class__.__name__) & (results_df['file_name'] == file_name)
                                        ].any().any()
                                    else:
                                        exists = results_df[
                                            (results_df["model_y"] ==
                                             model_y.__class__.__name__)
                                            & (results_df["model_t"] == model_t.__class__.__name__)
                                            & (results_df["causal_model_name"] == causal_model.__class__.__name__) & (results_df['file_name'] == file_name)
                                        ].any().any()
                                if exists:
                                    print(
                                        f"Skipping model_y: {model_y}, model_t: {model_t}, str_causal_model: {str_causal_model}")
                                    continue
                                temp_results, estimated_ite_values = causal_inference_analysis(
                                    model_y, model_t, causal_model, x_scaled, Y, T, true_ATE, true_ATE_stderr, true_ite, is_meta)
                                temp_results['data'] = key
                                temp_results['file_name'] = file_name
                                all_results.append(temp_results)
                                results_df = pd.DataFrame(all_results)

                                if i % 10 == 0:
                                    most_recent = results_df.tail(10)
                                    recent_10_table = wandb.Table(
                                        dataframe=most_recent)
                                    wandb.log(
                                        {"most_recent_10_scores_table": recent_10_table})
                                if i % 50 == 0:
                                    top_10_scores = results_df.groupby('causal_model_name').apply(
                                        lambda x: x.nsmallest(10, 'tao_risk')).reset_index(drop=True)
                                    top_10_scores_table = wandb.Table(
                                        dataframe=top_10_scores)
                                    wandb.log(
                                        {"top_10_scores_table": top_10_scores_table})
                                results_df.to_csv(
                                    f'results/{key}_no_params_baselines.csv')
                                print(
                                    f"Completed running model_y: {model_y}, model_t: {model_t}, str_causal_model: {str_causal_model}")
                            except Exception as e:
                                print(
                                    f"Error occurred while running {model_y}-{model_t} estimator with {str_causal_model} method: {str(e)}")
                            i += 1
                            count += 1
                results_df.to_csv(f"results/{key}_no_params_baselines.csv")
        else:
            data, X, T, Y, true_ite, true_ATE, true_ATE_stderr, is_discrete = data_dict[key]
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(X)
            results_file = f'results/{key}_no_params_baselines.csv'
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
            for str_causal_model in ci_estimators:
                is_meta = False
                if str_causal_model in ['sl', 'xl', 'tl']:
                    is_meta = True
                for model_y in my_list:
                    count = 0
                    for model_t in mt_list:
                        if is_meta and count >= 1:
                            continue
                        try:
                            causal_model = get_estimators(
                                str_causal_model, model_y, model_t)
                            exists = False
                            if os.path.exists(results_file):
                                if is_meta:
                                    exists = results_df[
                                        (results_df["model_y"] ==
                                         model_y.__class__.__name__)
                                        & (results_df["causal_model_name"] == causal_model.__class__.__name__)
                                    ].any().any()
                                else:
                                    exists = results_df[
                                        (results_df["model_y"] ==
                                         model_y.__class__.__name__)
                                        & (results_df["model_t"] == model_t.__class__.__name__)
                                        & (results_df["causal_model_name"] == causal_model.__class__.__name__)
                                    ].any().any()
                            if exists:
                                print(
                                    f"Skipping model_y: {model_y}, model_t: {model_t}, str_causal_model: {str_causal_model}")
                                continue
                            temp_results, estimated_ite_values = causal_inference_analysis(
                                model_y, model_t, causal_model, x_scaled, Y, T, true_ATE, true_ATE_stderr, true_ite, is_meta)
                            temp_results['data'] = key
                            all_results.append(temp_results)
                            results_df = pd.DataFrame(all_results)

                            if i % 10 == 0:
                                most_recent = results_df.tail(10)
                                recent_10_table = wandb.Table(
                                    dataframe=most_recent)
                                wandb.log(
                                    {"most_recent_10_scores_table": recent_10_table})
                            if i % 50 == 0:
                                top_10_scores = results_df.groupby('causal_model_name').apply(
                                    lambda x: x.nsmallest(10, 'tao_risk')).reset_index(drop=True)
                                top_10_scores_table = wandb.Table(
                                    dataframe=top_10_scores)
                                wandb.log(
                                    {"top_10_scores_table": top_10_scores_table})
                                results_df.to_csv(
                                    f'results/{key}_no_params_baselines.csv')
                            print(
                                f"Completed running model_y: {model_y}, model_t: {model_t}, str_causal_model: {str_causal_model}")
                        except Exception as e:
                            print(
                                f"Error occurred while running {model_y}-{model_t} estimator with {str_causal_model} method: {str(e)}")
                        i += 1
                        count += 1
            results_df.to_csv(f"results/{key}_no_params_baselines.csv")
        wandb.alert(title="Code is done!", text="Code is done!")
        wandb.finish()
