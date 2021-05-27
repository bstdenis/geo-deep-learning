import argparse
from pathlib import Path
import pickle
from functools import partial
import pprint

import mlflow
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from utils.readers import read_parameters
from train_segmentation import main as train_main

my_space = {'target_size': hp.choice('target_size', [128, 256]),
            'model_name': hp.choice('model_name', ['unet', 'deeplabv3+_pretrained']),
            'permanent_water_weight': hp.uniform('permanent_water_weight', 1.0, 10.0),
            'rivers_weight': hp.uniform('rivers_weight', 1.0, 10.0),
            'flood_weight': hp.uniform('flood_weight', 1.0, 10.0),
            'noise': hp.choice('noise', [0.0, 1.0])}


def get_latest_mlrun(params):
    tracking_uri = params['global']['mlflow_uri']
    mlflow.set_tracking_uri(tracking_uri)
    mlexp = mlflow.get_experiment_by_name(params['global']['mlflow_experiment_name'])
    exp_id = mlexp.experiment_id
    try:
        run_ids = ([x.run_id for x in mlflow.list_run_infos(
            exp_id, max_results=1, order_by=["tag.release DESC"])])
    except AttributeError:
        mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        run_ids = [x.run_id for x in mlflow_client.list_run_infos(exp_id, run_view_type=3)[0:1]]
    mlrun = mlflow.get_run(run_ids[0])
    return mlrun


def objective_with_args(a, params, config_path):
    params['training']['target_size'] = a['target_size']
    params['global']['model_name'] = a['model_name']
    # ToDo should adjust batch size as a function of model and target size...
    params['training']['class_weights'] = [1.0, a['permanent_water_weight'], a['rivers_weight'], a['flood_weight']]
    params['training']['augmentation']['noise'] = a['noise']

    try:
        mlrun = get_latest_mlrun(params)
        run_name_split = mlrun.data.tags['mlflow.runName'].split('_')
        params['global']['mlflow_run_name'] = run_name_split[0] + f'_{int(run_name_split[1])+1}'
    except:
        pass

    train_main(params, config_path)

    mlflow.end_run()
    mlrun = get_latest_mlrun(params)

    # ToDo Probably need some cleanup to avoid accumulating results on disk

    return {'loss': -mlrun.data.metrics['tst_iou_nonbg'], 'status': STATUS_OK}



def main(params, config_path):
    if Path('hyperopt_trials.pkl').is_file():
        trials = pickle.load(open("hyperopt_trials.pkl", "rb"))
    else:
        trials = Trials()

    objective = partial(objective_with_args, params=params, config_path=config_path)

    n = 0

    while n < params['global']['hyperopt_runs']:
        best = fmin(objective,
                    space=my_space,
                    algo=tpe.suggest,
                    trials=trials,
                    max_evals=n+params['global']['hyperopt_delta'])
        n += params['global']['hyperopt_delta']
        pickle.dump(trials, open("hyperopt_trials.pkl", "wb"))

    pprint.pprint(trials.vals)
    pprint.pprint(trials.results)
    for key, val in best.items():
        if my_space[key].name == 'switch':
            best[key] = my_space[key].pos_args[val+1].obj
    pprint.pprint(best)
    print(trials.best_trial['result'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Geo Deep Learning hyperopt')
    parser.add_argument('param_file', type=str, help='Path of gdl config file')
    args = parser.parse_args()
    config_path = Path(args.param_file)
    params = read_parameters(args.param_file)
    params['self'] = {'config_file': args.param_file}
    main(params, config_path)
