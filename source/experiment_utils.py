import tensorflow as tf
from general_utils import gpu_session, get_mlp_server, aulc, new_session
from data_utils import *
from network_manager import NetworkManager
import numpy as np
from collections import defaultdict


def run_network_manager(sequence, x, y, test_data, data_set, model, training, _run, method,
                        validation_data=None):
    network_m = NetworkManager(get_mlp_server(model['input_shape'], model['layer'],
                                                          model['layer_units'], model['activations'],
                                                          data_set['data_set_size'][0], model['num_samples'], model['dropout']),
                               data_set_size=data_set['data_set_size'], n_samples=model['num_samples'],
                               num_clients=data_set['num_tasks'], run_obj=_run, method=method)
    network_m.compile(optimizer=training['optimizer'](training['learning_rate'], training['decay']),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=[tf.keras.metrics.categorical_accuracy])
    fit_args = {'sequence': sequence,
                'x': x,
                'y': y,
                'validation_split': training['validation_split'],
                'validation_data': validation_data,
                'epochs': training['num_epochs'],
                'test_data': test_data,
                'batch_size': training['batch_size'],
                'verbose': training['verbose']
                }
    if training['early_stopping']:
        fit_args['callbacks'] = [training['early_stopping']]

    return network_m.fit(**fit_args)


def simulation(method, x, y, test_data, data_set, model, training, _run, validation_data=None):
    if method == 'virtual':
        sequence = np.tile(range(data_set['num_tasks']), training['num_refining'])
        return run_network_manager(sequence, x, y, test_data, data_set, model, training, _run,
                                   method, validation_data)
    if method == 'local':
        history_local = defaultdict(list)
        evaluate_local = defaultdict(list)
        for n in range(data_set['num_tasks']):
            print('local model number ', n + 1)
            sequence = [n]
            hist, eval = run_network_manager(sequence, x, y, test_data, data_set, model,
                                             training, _run, method, validation_data)
            history_local[n].append(hist.values())
            evaluate_local[n].append(np.array(list(eval.values())).squeeze())
            print(evaluate_local[n])
            sess = new_session(sess_config=config)

        return history_local, evaluate_local
    if method == 'global':
        x = np.concatenate(x)
        y = np.concatenate(y)
        x_t, y_t = zip(*validation_data)
        x_t = np.concatenate(x_t)
        y_t = np.concatenate(y_t)
        sequence = [0]
        return run_network_manager(sequence, [x], [y], [(x_t, y_t)], data_set, model,
                                   training, _run, method, validation_data)
