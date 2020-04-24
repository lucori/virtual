from os import environ as os_environ
import subprocess
from shutil import copytree
import datetime
from itertools import product
from pathlib import Path
import argparse

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import json
import gc

from source.data_utils import federated_dataset, batch_dataset
from source.utils import gpu_session
from source.experiment_utils import (run_simulation,
                                     get_compiled_model_fn_from_dict)


def create_hparams(hp_conf, data_set_conf, training_conf,
                   model_conf, logdir):

    # TODO: add a large predefined range for the most common parameters of
    #  grid search. This makes the tensorboard search possible.
    HP_DICT = {}
    for key_0, _ in hp_conf.items():
        HP_DICT[key_0] = hp.HParam(key_0)
    for key, _ in data_set_conf.items():
        HP_DICT[f'data_{key}'] = hp.HParam(f'data_{key}')
    for key, _ in training_conf.items():
        HP_DICT[f'training_{key}'] = hp.HParam(f'training_{key}')
    for key, _ in model_conf.items():
        HP_DICT[f'model_{key}'] = hp.HParam(f'model_{key}')
    HP_DICT['run'] = hp.HParam('run')
    HP_DICT['config_name'] = hp.HParam('config_name')

    metrics = [hp.Metric('sparse_categorical_accuracy',
                         display_name='Accuracy')]
    with tf.summary.create_file_writer(str(logdir)).as_default():
        hp.hparams_config(hparams=HP_DICT.values(),
                          metrics=metrics)
    return HP_DICT


def write_hparams(hp_dict, session_num, exp_conf, data_set_conf,
                  training_conf, model_conf, logdir_run, config_name):

    hparams = {'run': int(session_num), 'config_name': config_name}
    for key_0, value_0 in exp_conf.items():
        hparams[hp_dict[key_0]] = value_0
    for key_1, value_1 in data_set_conf.items():
        hparams[hp_dict[f'data_{key_1}']] = str(value_1)
    for key_2, value_2 in training_conf.items():
        hparams[hp_dict[f'training_{key_2}']] = str(value_2)
    for key_3, value_3 in model_conf.items():
        if key_3 == 'layers':
            continue
        hparams[hp_dict[f'model_{key_3}']] = str(value_3)

    # Only concatenation of the name of the layers
    layers = ''
    for layer in model_conf['layers']:
        layers = layers + layer['name'] + '_'
    hparams[hp_dict['model_layers']] = layers[:-1]

    with tf.summary.create_file_writer(str(logdir_run)).as_default():
        hp.hparams(hparams)


def _gridsearch(hp_conf):
    keys, values = zip(*hp_conf.items())
    experiments = [dict(zip(keys, v)) for v in product(*values)]
    return experiments


def submit_jobs(configs, root_path, data_dir, mem=8000, use_scratch=False):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_dir = root_path / f'temp_configs_{current_time}'
    config_dir.mkdir(exist_ok=True)

    hp_conf = configs['hp']
    experiments = _gridsearch(hp_conf)
    new_config = configs.copy()
    for session_num, exp_conf in enumerate(experiments):

        for key, value in exp_conf.items():
            new_config['hp'][key] = [value]

        # Save the new config file
        config_path = config_dir / f"{configs['config_name']}_" \
                                   f"g{current_time}_" \
                                   f"{session_num}.json"
        with config_path.open(mode='w') as config_file:
            json.dump(new_config, config_file)

        # Run training with the new config file
        command = (f"bsub -n 2 -W 12:00 "
                   f"-R rusage[mem={mem},scratch=80000,"
                   f"ngpus_excl_p=1] "
                   f"python main.py --result_dir {root_path} "
                   f"--data_dir {data_dir} "
                   f"{'--scratch ' if use_scratch else ''}"
                   f"{config_path}")
        subprocess.check_output(command.split())


def run_experiments(configs, root_path, data_dir=None, use_scratch=False):

    if use_scratch:
        dir_name = data_dir.name
        temp_dir = Path(os_environ['TMPDIR']) / dir_name
        print(f"Copying datafiles to the scratch folder {temp_dir}")
        copytree(str(data_dir), str(temp_dir))
        data_dir = temp_dir

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Configs
    data_set_conf = configs['data_set_conf']
    training_conf = configs['training_conf']
    model_conf = configs['model_conf']
    hp_conf = configs['hp']
    if 'input_shape' in model_conf:
        model_conf['input_shape'] = tuple(model_conf['input_shape'])
    logdir = root_path / 'logs' / f'{configs["config_name"]}_' \
                                  f'e{current_time}'

    fede_train_data, fed_test_data, train_size, test_size = federated_dataset(
        data_set_conf, data_dir)
    num_clients = len(fede_train_data)
    model_conf['num_clients'] = num_clients

    HP_DICT = create_hparams(hp_conf, data_set_conf, training_conf,
                             model_conf, logdir)

    experiments = _gridsearch(hp_conf)
    for session_num, exp_conf in enumerate(experiments):
        all_params = {**data_set_conf, **training_conf, **model_conf,
                      **exp_conf}

        seq_length = data_set_conf.get('seq_length', None)
        federated_train_data_batched = [
            batch_dataset(data, all_params['batch_size'],
                          padding=data_set_conf['name'] == 'shakespeare',
                          seq_length=seq_length)
            for data in fede_train_data]
        federated_test_data_batched = [
            batch_dataset(data, all_params['batch_size'],
                          padding=data_set_conf['name'] == 'shakespeare',
                          seq_length=seq_length)
            for data in fed_test_data]

        sample_batch = tf.nest.map_structure(
            lambda x: x.numpy(), iter(federated_train_data_batched[0]).next())

        logdir_run = logdir / f'{session_num}_{current_time}'
        print(f"saving results in {logdir_run}")
        write_hparams(HP_DICT, session_num, exp_conf, data_set_conf,
                      training_conf, model_conf, logdir_run, configs[
                          'config_name'])
        with open(logdir_run / 'config.json', 'w') as config_file:
            json.dump(configs, config_file, indent=4)

        print(f'Starting run {session_num} with parameters {all_params}...')
        model_fn = get_compiled_model_fn_from_dict(all_params, sample_batch)
        run_simulation(model_fn, federated_train_data_batched,
                       federated_test_data_batched, train_size, test_size,
                       all_params, logdir_run)
        tf.keras.backend.clear_session()
        gc.collect()


def main():
    # Parse arguments
    # TODO: Add a logging system instead of prints.

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path",
                        type=Path,
                        help="Path to the main json config. "
                             "Ex: 'configurations/femnist_virtual.json'")
    parser.add_argument("--result_dir",
                        type=Path,
                        help="Path in which results of training are/will be "
                             "located")
    parser.add_argument("--data_dir",
                        type=Path,
                        help="Path in which data is located. This is "
                             "required if run on Leonhard")
    parser.add_argument("--submit_leonhard", action='store_true',
                        help="Whether to submit jobs to leonhard for "
                             "grid search")

    parser.add_argument("--scratch", action='store_true',
                        help="Whether to first copy the dataset to the "
                             "scratch storage of Leonhard. Do not use on "
                             "other systems than Leonhard.")
    parser.add_argument("--mem",
                        type=int,
                        default=4500,
                        help="Memory allocated for each leonhard job. This "
                             "will be ignored of Leonhard is not selected.")

    args = parser.parse_args()
    # Read config files
    with args.config_path.absolute().open(mode='r') as config_file:
        configs = json.load(config_file)
        configs['config_name'] = args.config_path.name.\
            replace(args.config_path.suffix, "")

    if not args.result_dir:
        args.result_dir = Path(__file__).parent.absolute().parent

    if args.scratch and not args.data_dir:
        print("WARNING: You can not use scratch while not giving the "
              "datafolder. Scratch will be ignored.")
        args.scratch = False

    if args.submit_leonhard:
        submit_jobs(configs, args.result_dir, args.data_dir, args.mem,
                    args.scratch)
    else:
        gpu_session(configs['session']['num_gpus'])
        run_experiments(configs, args.result_dir, args.data_dir, args.scratch)


if __name__ == "__main__":
    main()
