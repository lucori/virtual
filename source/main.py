import os
import sys
import datetime
from itertools import product

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import json
import gc

from source.data_utils import federated_dataset, batch_dataset
from source.utils import gpu_session
from source.experiment_utils import (run_simulation,
                                     get_compiled_model_fn_from_dict)


dir_path = os.path.dirname(os.path.realpath(__file__))

# current time for file names
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print("Time:", current_time)

if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = 'configurations/femnist_virtual.json'

with open(dir_path + '/../' + config_file) as config_file:
    config = json.load(config_file)

session_conf = config['session']
data_set_conf = config['data_set_conf']
training_conf = config['training_conf']
model_conf = config['model_conf']
if 'input_shape' in model_conf:
    model_conf['input_shape'] = tuple(model_conf['input_shape'])
hp_conf = config['hp']

gpu_session(session_conf['num_gpus'])

federated_train_data, federated_test_data, train_size, test_size = federated_dataset(data_set_conf)

num_clients = len(federated_train_data)
model_conf['num_clients'] = num_clients

HP_DICT = {}
runs = 1
for key, values in hp_conf.items():
    HP_DICT[key] = hp.HParam(key,  hp.Discrete(values))
    runs = runs*len(values)
HP_DICT['run'] = hp.HParam('run', hp.Discrete(range(runs)))
HP_DICT['method'] = hp.HParam('method', hp.Discrete(['virtual', 'fedprox']))

logdir = os.path.join(dir_path, '../logs', data_set_conf['name'], training_conf['method'], current_time)

with tf.summary.create_file_writer(logdir).as_default():
    hp.hparams_config(hparams=HP_DICT.values(),
                      metrics=[hp.Metric('sparse_categorical_accuracy', display_name='Accuracy')],
                      )


keys, values = zip(*hp_conf.items())
experiments = [dict(zip(keys, v)) for v in product(*values)]
seq_length = data_set_conf.get('seq_length', None)

for session_num, exp in enumerate(experiments):
    #TODO: add all parameters to hp so that multiple runs can be sorted together.
    #TODO: use multithreding or solve OOM problem with multiple runs in one gpu
    all_params = {**data_set_conf, **training_conf, **model_conf, **exp}
    federated_train_data_batched = [batch_dataset(data, all_params['batch_size'],
                                                  padding=data_set_conf['name'] == 'shakespeare',
                                                  seq_length=seq_length)
                                    for data in federated_train_data]
    federated_test_data_batched = [batch_dataset(data, all_params['batch_size'],
                                                 padding=data_set_conf['name'] == 'shakespeare',
                                                 seq_length=seq_length)
                                   for data in federated_test_data]

    sample_batch = tf.nest.map_structure(
        lambda x: x.numpy(), iter(federated_train_data_batched[0]).next())

    hparams = dict([(HP_DICT[key], value) for key, value in exp.items()])
    hparams['run'] = session_num
    hparams['method'] = all_params['method']

    logdir_run = logdir + '/' + str(session_num)
    with tf.summary.create_file_writer(logdir_run).as_default():
        hp.hparams(hparams)

    model_fn = get_compiled_model_fn_from_dict(all_params, sample_batch)
    run_simulation(model_fn, federated_train_data_batched, federated_test_data_batched, train_size, test_size,
                   all_params, logdir_run)
    tf.keras.backend.clear_session()
    gc.collect()
