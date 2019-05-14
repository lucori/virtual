import tensorflow as tf
from general_utils import gpu_session, get_mlp_server, aulc
from data_utils import *
from layers import DenseReparameterizationPriorUpdate
from network_manager import NetworkManager
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds


ex = Experiment("experiments")

ex.observers.append(
    FileStorageObserver.create("runs")
)

ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    session = {
        "num_gpus": 0,
        "gpus": None,
        "experiment_name": None,
        "num_runs": 0
    }
    data_set = {
        "num_tasks": 0,
        "min_data_set_size_per_user": 0,
        "max_data_set_size_per_user": None,
        "test_size": 0,
        "generator": None
    }
    training = {
        "num_epochs": 0,
        "batch_size": 0,
        "num_refining": 0,
        "patience": 0,
        "learning_rate": 0,
        "validation_split": 0,
        "verbose": 0,
        "early_stopping": False
    }
    model = {
        "num_samples": 0,
        "input_shape": [],
        "layer": None,
        "layer_units": [],
        "activations": []
    }


@ex.capture
def get_globals(data_set, model):
    model['layer'] = globals()[model['layer']]
    data_set['generator'] = globals()[data_set['generator']]


@ex.automain
def run(session, data_set, training, model):
    get_globals()
    config, distribution = gpu_session(num_gpus=session['num_gpus'], gpus=session['gpus'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=training['patience'],
                                                      restore_best_weights=True)
    if config:
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)

    history_local_multi_runs = []
    history_virtual_multi_runs = []
    history_global_multi_runs = []
    evaluation_virtual_multi_runs = []
    evaluation_local_multi_runs = []
    evaluation_global_multi_runs = []
    local_average_multi_runs = []
    virtual_average_multi_runs = []
    aulcs_multi_runs = []
    normalized_accuracy_multi_runs = []

    for i in range(1, session['num_runs']+1):
        print('run: ', i, 'out of: ', session['num_runs'])

        x, y, x_t, y_t = data_set['generator'](data_set['num_tasks'],
                                               min_data_set_size_per_user=data_set['min_data_set_size_per_user'],
                                               max_data_set_size_per_user=data_set['max_data_set_size_per_user'],
                                               test_size=data_set['test_size'])
        data_set_size = [x_i.shape[0] for x_i in x]
        print(data_set_size)
        data_set['num_tasks'] = len(data_set_size)
        network_m = NetworkManager(lambda dt_size: get_mlp_server(model['input_shape'], model['layer'],
                                                                  model['layer_units'], model['activations'],
                                                                  dt_size, model['num_samples']),
                                   data_set_size=data_set_size, n_samples=model['num_samples'],
                                   num_clients=data_set['num_tasks'], sess_config=config)
        network_m.compile(optimizer=tf.keras.optimizers.Adam(training['learning_rate']),
                          loss=tf.keras.losses.categorical_crossentropy,
                          metrics=[tf.keras.metrics.categorical_accuracy])
        sequence = np.tile(range(data_set['num_tasks']), training['num_refining'])
        fit_args_virtual = {'sequence': sequence,
                            'x': x,
                            'y': y,
                            'validation_split': training['validation_split'],
                            'validation_data': zip(x_t, y_t),
                            'epochs': training['num_epochs'],
                            'test_data': zip(x_t, y_t),
                            'batch_size': training['batch_size'],
                            'verbose': training['verbose']
                            }
        if training['early_stopping']:
            fit_args_virtual['callbacks'] = [early_stopping]
        history_virtual, evaluation_virtual = network_m.fit(**fit_args_virtual)
        print('virtual done')

        tf.reset_default_graph()
        if config:
            sess = tf.Session(config=config)
            tf.keras.backend.set_session(sess)

        history_local = {}
        evaluation_local = {}
        for n in range(data_set['num_tasks']):
            print('local model number ', n+1)
            network_m = NetworkManager(lambda dt_size: get_mlp_server(model['input_shape'], model['layer'],
                                                                      model['layer_units'], model['activations'],
                                                                      dt_size, model['num_samples']),
                                       data_set_size=data_set_size, n_samples=model['num_samples'],
                                       num_clients=data_set['num_tasks'], sess_config=config)
            network_m.compile(optimizer=tf.keras.optimizers.Adam(training['learning_rate']),
                              loss=tf.keras.losses.categorical_crossentropy,
                              metrics=[tf.keras.metrics.categorical_accuracy])
            sequence = [n]
            fit_args_local = {'sequence': sequence,
                                'x': x,
                                'y': y,
                                'validation_split': training['validation_split'],
                                'validation_data': zip(x_t, y_t),
                                'epochs': training['num_epochs'],
                                'test_data': zip(x_t, y_t),
                                'batch_size': training['batch_size'],
                                'verbose': training['verbose']
                                }
            if training['early_stopping']:
                fit_args_local['callbacks'] = [early_stopping]
            history_local, evaluation_local = network_m.fit(**fit_args_local)
            print(evaluation_local[n])
            tf.reset_default_graph()
            if config:
                sess = tf.Session(config=config)
                tf.keras.backend.set_session(sess)

        x = np.concatenate(x)
        y = np.concatenate(y)
        x_t = np.concatenate(x_t)
        y_t = np.concatenate(y_t)

        network_m = NetworkManager(lambda dt_size: get_mlp_server(model['input_shape'], model['layer'],
                                                                  model['layer_units'], model['activations'],
                                                                  dt_size, model['num_samples']),
                                   data_set_size=[sum(data_set_size)], n_samples=model['num_samples'],
                                   num_clients=1, sess_config=config)
        network_m.compile(optimizer=tf.keras.optimizers.Adam(training['learning_rate']),
                          loss=tf.keras.losses.categorical_crossentropy,
                          metrics=[tf.keras.metrics.categorical_accuracy])
        sequence = [0]
        fit_args_global = {'sequence': sequence,
                           'x': [x],
                           'y': [y],
                           'validation_split': training['validation_split'],
                           'validation_data': [(x_t, y_t)],
                           'epochs': training['num_epochs'],
                           'test_data': [(x_t, y_t)],
                           'batch_size': training['batch_size'],
                           'verbose': training['verbose']
                           }
        if training['early_stopping']:
            fit_args_global['callbacks'] = [early_stopping]
        history_global, evaluation_global = network_m.fit(**fit_args_global)
        print(evaluation_global)

        local_average = np.array(list(evaluation_local.values())).mean(axis=0)
        virtual_average = np.array(list(evaluation_virtual.values())).mean(axis=(0, 1))
        normalized_accuracy = np.array([(np.array(v)/np.array(l)) for (v, l) in
                                       zip(list(evaluation_virtual.values()),
                                           list(evaluation_local.values()))])[:, :, 1]
        aulcs = aulc(history_virtual)

        history_virtual_multi_runs.append(history_virtual)
        history_local_multi_runs.append(history_local)
        history_global_multi_runs.append(history_global)
        evaluation_virtual_multi_runs.append(evaluation_virtual)
        evaluation_local_multi_runs.append(evaluation_local)
        evaluation_global_multi_runs.append(evaluation_global)
        local_average_multi_runs.append(local_average)
        virtual_average_multi_runs.append(virtual_average)
        aulcs_multi_runs.append(aulcs)
        normalized_accuracy_multi_runs.append(normalized_accuracy)

    ex.info['virtual_history'] = history_virtual_multi_runs
    ex.info['virtual_evaluation'] = evaluation_virtual_multi_runs
    ex.info['local_history'] = history_local_multi_runs
    ex.info['local_evaluation'] = evaluation_local_multi_runs
    ex.info['global_history'] = history_global_multi_runs
    ex.info['global_evaluation'] = evaluation_global_multi_runs
    ex.info['local_average'] = local_average_multi_runs
    ex.info['virtual_average'] = virtual_average_multi_runs
    ex.info['aulc'] = aulcs_multi_runs
    ex.info['normalized_virtual_accuracy'] = normalized_accuracy_multi_runs

    print('done')
    return evaluation_virtual_multi_runs, evaluation_local_multi_runs, evaluation_global_multi_runs
