import sys
import datetime
from itertools import product
from pathlib import Path

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

    metrics = [hp.Metric('sparse_categorical_accuracy',
                         display_name='Accuracy')]
    with tf.summary.create_file_writer(str(logdir)).as_default():
        hp.hparams_config(hparams=HP_DICT.values(),
                          metrics=metrics)
    return HP_DICT


def write_hparams(hp_dict, session_num, exp_conf, data_set_conf,
                  training_conf, model_conf, logdir_run):

    hparams = {'run': int(session_num)}
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


# Paths
file_path = Path(__file__).parent.absolute()
root_path = file_path.parent
if len(sys.argv) > 1:
    config_path = Path(sys.argv[1])
else:
    config_path = Path('configurations/femnist_virtual.json')
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Configs
with open(root_path / config_path) as config_file:
    config = json.load(config_file)
session_conf = config['session']
data_set_conf = config['data_set_conf']
training_conf = config['training_conf']
model_conf = config['model_conf']
hp_conf = config['hp']
if 'input_shape' in model_conf:
    model_conf['input_shape'] = tuple(model_conf['input_shape'])

logdir = root_path / 'logs' / f'{data_set_conf["name"]}_' \
                              f'{training_conf["method"]}_' \
                              f'{current_time}'


gpu_session(session_conf['num_gpus'])

fede_train_data, fed_test_data, train_size, test_size = federated_dataset(data_set_conf)

num_clients = len(fede_train_data)
model_conf['num_clients'] = num_clients

HP_DICT = create_hparams(hp_conf, data_set_conf, training_conf,
                         model_conf, logdir)


keys, values = zip(*hp_conf.items())
experiments = [dict(zip(keys, v)) for v in product(*values)]
seq_length = data_set_conf.get('seq_length', None)

for session_num, exp_conf in enumerate(experiments):
    # TODO: use multithreding or solve OOM problem with multiple runs in one gpu
    # TODO: Maybe make grid search on all parameters easier

    all_params = {**data_set_conf, **training_conf, **model_conf, **exp_conf}
    federated_train_data_batched = [batch_dataset(data, all_params['batch_size'],
                                                  padding=data_set_conf['name'] == 'shakespeare',
                                                  seq_length=seq_length)
                                    for data in fede_train_data]
    federated_test_data_batched = [batch_dataset(data, all_params['batch_size'],
                                                 padding=data_set_conf['name'] == 'shakespeare',
                                                 seq_length=seq_length)
                                   for data in fed_test_data]

    sample_batch = tf.nest.map_structure(
        lambda x: x.numpy(), iter(federated_train_data_batched[0]).next())

    logdir_run = logdir / f'{session_num}_{current_time}'
    print(f"saving results in {logdir_run}")
    write_hparams(HP_DICT, session_num, exp_conf, data_set_conf,
                  training_conf, model_conf, logdir_run)
    with open(logdir_run / 'config.json', 'w') as config_file:
        json.dump(config, config_file, indent=4)

    print(f'Starting run {session_num} with parameters {all_params}...')
    model_fn = get_compiled_model_fn_from_dict(all_params, sample_batch)
    run_simulation(model_fn, federated_train_data_batched, federated_test_data_batched, train_size, test_size,
                   all_params, logdir_run)
    tf.keras.backend.clear_session()
    gc.collect()



