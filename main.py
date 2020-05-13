from os import environ as os_environ
import sys
from git import Repo
import subprocess
from shutil import copytree
import datetime
from itertools import product
from pathlib import Path
import argparse
import logging

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import json
import gc

from source.data_utils import federated_dataset, batch_dataset
from source.utils import gpu_session
from source.experiment_utils import (run_simulation,
                                     get_compiled_model_fn_from_dict)
from source.constants import ROOT_LOGGER_STR, LOGGER_RESULT_FILE


logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


def _setup_logger(results_path, create_stdlog):
    """Setup a general logger which saves all logs in the experiment folder"""

    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler = logging.FileHandler(str(results_path))
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(f_format)

    root_logger = logging.getLogger(ROOT_LOGGER_STR)
    root_logger.handlers = []
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(f_handler)

    if create_stdlog:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        root_logger.addHandler(handler)


def create_hparams(hp_conf, data_set_conf, training_conf,
                   model_conf, logdir):
    exp_wide_keys = ["learning_rate", "l2_reg", "kl_weight", "batch_size",
                     "epochs_per_round", "hierarchical", "prior_scale"]
    HP_DICT = {}
    for key_0 in list(hp_conf.keys()) + exp_wide_keys:
        if (key_0 == 'learning_rate'
                or key_0 == 'kl_weight'
                or key_0 == 'l2_reg'):
            HP_DICT[key_0] = hp.HParam(key_0, hp.RealInterval(0.0, 1.0))
        elif key_0 == 'batch_size':
            HP_DICT[key_0] = hp.HParam(key_0, hp.Discrete([1, 5, 10, 20, 40, 50,
                                                           64, 128, 256, 512]))
        elif key_0 == 'epochs_per_round':
            HP_DICT[key_0] = hp.HParam(key_0, hp.Discrete([1, 5, 10, 15, 20,
                                                           25, 30, 35, 40,
                                                           45, 50, 55, 60,
                                                           65, 70, 75, 80,
                                                           85, 90, 95, 100,
                                                           110, 120, 130,
                                                           140, 150]))
        elif key_0 == 'clients_per_round':
            HP_DICT[key_0] = hp.HParam(key_0, hp.Discrete([1, 2, 3, 4, 5,
                                                           10, 15, 20, 50]))
        elif key_0 == 'method':
            HP_DICT[key_0] = hp.HParam(key_0, hp.Discrete(['virtual',
                                                           'fedprox']))
        elif key_0 == 'hierarchical':
            HP_DICT[key_0] = hp.HParam(key_0, hp.Discrete([True, False]))
        else:
            HP_DICT[key_0] = hp.HParam(key_0)
    for key, _ in data_set_conf.items():
        if key == 'name':
            HP_DICT[f'data_{key}'] = hp.HParam(f'data_{key}',
                                               hp.Discrete(['mnist',
                                                            'pmnist',
                                                            'femnist',
                                                            'shakespeare',
                                                            'human_activity',
                                                            'vehicle_sensor']))
        else:
            HP_DICT[f'data_{key}'] = hp.HParam(f'data_{key}')
    for key, _ in training_conf.items():
        HP_DICT[f'training_{key}'] = hp.HParam(f'training_{key}')
    for key, _ in model_conf.items():
        HP_DICT[f'model_{key}'] = hp.HParam(f'model_{key}')
    HP_DICT['run'] = hp.HParam('run')
    HP_DICT['config_name'] = hp.HParam('config_name')
    HP_DICT['training_num_rounds'] = hp.HParam('num_rounds',
                                               hp.RealInterval(0.0, 1e10))

    metrics = [hp.Metric('sparse_categorical_accuracy',
                         display_name='Accuracy'),
               hp.Metric('max_sparse_categorical_accuracy',
                         display_name='Max Accuracy')]
    with tf.summary.create_file_writer(str(logdir)).as_default():
        hp.hparams_config(hparams=HP_DICT.values(),
                          metrics=metrics)
    return HP_DICT


def write_hparams(hp_dict, session_num, exp_conf, data_set_conf,
                  training_conf, model_conf, logdir_run, config_name):

    hparams = {'run': int(session_num), 'config_name': config_name}
    for key_0, value_0 in exp_conf.items():
        if isinstance(value_0, list):
            hparams[hp_dict[key_0]] = str(value_0)
        else:
            hparams[hp_dict[key_0]] = value_0
    for key_1, value_1 in data_set_conf.items():
        hparams[hp_dict[f'data_{key_1}']] = str(value_1)
    for key_2, value_2 in training_conf.items():
        if key_2 == 'num_rounds':
            hparams[hp_dict[f'training_{key_2}']] = value_2
        else:
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


def submit_jobs(configs, root_path, data_dir, hour=12, mem=8000,
                use_scratch=False, reps=1):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_dir = root_path / f'temp_configs_{current_time}'
    config_dir.mkdir(exist_ok=True)

    hp_conf = configs['hp']
    experiments = _gridsearch(hp_conf)
    new_config = configs.copy()
    for session_num, exp_conf in enumerate(experiments):
        for rep in range(reps):
            for key, value in exp_conf.items():
                new_config['hp'][key] = [value]

            # Save the new config file
            config_path = config_dir / f"{configs['config_name']}_" \
                                       f"g{current_time}_" \
                                       f"{session_num}_" \
                                       f"{rep}.json"
            with config_path.open(mode='w') as config_file:
                json.dump(new_config, config_file)

            # Run training with the new config file
            command = (f"bsub -n 2 -W {hour}:00 "
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
        logger.info(f"Copying datafiles to the scratch folder {temp_dir}")
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
    logdir.mkdir(parents=True)
    logfile = logdir / LOGGER_RESULT_FILE
    _setup_logger(logfile, create_stdlog=True)
    commit_id = Repo(Path().absolute()).head.commit
    logger.debug(f"Running code on git commit {commit_id}")

    fede_train_data, fed_test_data, train_size, test_size = federated_dataset(
        data_set_conf, data_dir)
    num_clients = len(fede_train_data)
    model_conf['num_clients'] = num_clients

    experiments = _gridsearch(hp_conf)
    for session_num, exp_conf in enumerate(experiments):
        all_params = {**data_set_conf,
                      **training_conf,
                      **model_conf,
                      **exp_conf}

        training_conf['num_rounds'] = \
            int(all_params['tot_epochs_per_client']
                / (all_params['clients_per_round']
                   * all_params['epochs_per_round']))
        all_params['num_rounds'] = training_conf['num_rounds']

        # Log configurations
        logdir_run = logdir / f'{session_num}_{current_time}'
        logger.info(f"saving results in {logdir_run}")
        HP_DICT = create_hparams(hp_conf, data_set_conf, training_conf,
                                 model_conf, logdir_run)

        write_hparams(HP_DICT, session_num, exp_conf, data_set_conf,
                      training_conf, model_conf, logdir_run, configs[
                          'config_name'])

        with open(logdir_run / 'config.json', 'w') as config_file:
            json.dump(configs, config_file, indent=4)

        # Prepare dataset
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

        # Run the experiment
        logger.info(f'Starting run {session_num} '
                    f'with parameters {all_params}...')
        model_fn = get_compiled_model_fn_from_dict(all_params, sample_batch)
        run_simulation(model_fn, federated_train_data_batched,
                       federated_test_data_batched, train_size, test_size,
                       all_params, logdir_run)
        tf.keras.backend.clear_session()
        gc.collect()

        logger.info("Finished experiment successfully")


def main():
    # Parse arguments

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
                        default=Path('data'),
                        help="Path in which data is located. This is "
                             "required if run on Leonhard")
    parser.add_argument("--submit_leonhard", action='store_true',
                        help="Whether to submit jobs to leonhard for "
                             "grid search")

    parser.add_argument("-s", "--scratch", action='store_true',
                        help="Whether to first copy the dataset to the "
                             "scratch storage of Leonhard. Do not use on "
                             "other systems than Leonhard.")
    parser.add_argument("-m", "--memory",
                        type=int,
                        default=8500,
                        help="Memory allocated for each leonhard job. This "
                             "will be ignored of Leonhard is not selected.")
    parser.add_argument("-t", "--time",
                        type=int,
                        default=24,
                        help="Number of hours requested for the job on "
                             "Leonhard. For virtual models usually it "
                             "requires more time than this default value.")
    parser.add_argument("-r", "--repetitions",
                        type=int,
                        default=1,
                        help="Number of repetitions to run the same "
                             "experiment")

    args = parser.parse_args()
    # Read config files
    with args.config_path.absolute().open(mode='r') as config_file:
        configs = json.load(config_file)
        configs['config_name'] = args.config_path.name.\
            replace(args.config_path.suffix, "")

    if not args.result_dir:
        args.result_dir = Path(__file__).parent.absolute()

    if args.scratch and args.data_dir == Path('data'):
        logger.warning("WARNING: You can not use scratch while not on "
                       "Leonhard. Make sure you understand what you are "
                       "doing.")

    if args.submit_leonhard:
        submit_jobs(configs, args.result_dir, args.data_dir,
                    hour=args.time, mem=args.memory, use_scratch=args.scratch,
                    reps=args.repetitions)
    else:
        gpu_session(configs['session']['num_gpus'])
        run_experiments(configs, args.result_dir, args.data_dir, args.scratch)


if __name__ == "__main__":
    main()
