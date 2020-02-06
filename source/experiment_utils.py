import os
import tensorflow as tf
from dense_reparametrization_shared import DenseReparametrizationShared
from tensorflow.keras.layers import Dense
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
import tensorflow_federated as tff
import datetime
from virtual_process import VirtualFedProcess
import random
import math
from fed_prox import FedProx
from centered_l2_regularizer import DenseCentered, CenteredL2Regularizer

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_compiled_model_fn_from_dict(model_dict, training_dict, sample_batch):
    layer = globals()[model_dict['layer']]

    def create_model(model_class=tf.keras.Sequential, train_size=None):
        args = {}
        if layer == DenseReparametrizationShared:
            kernel_divergence_fn = (lambda q, p, ignore:  model_dict['kl_weight'] * kl_lib.kl_divergence(q, p) / float(train_size))
            args['kernel_divergence_fn'] = kernel_divergence_fn
            args['num_clients'] = model_dict['num_clients']
            args['prior_scale'] = model_dict['prior_scale']
        if layer == DenseCentered:
            args['kernel_regularizer'] = lambda: CenteredL2Regularizer(model_dict['l2_reg'])
            args['bias_regularizer'] = lambda: CenteredL2Regularizer(model_dict['l2_reg'])
        layers = []
        for i, (l_u, act) in enumerate(zip(model_dict['layer_units'], model_dict['activations'])):
            if i == 0:
                args['input_shape'] = model_dict['input_shape']
            args['activation'] = act
            layers.append(layer(l_u, **args))
            args.pop('input_shape', None)

        return model_class(layers)

    def compile_model(model):

        def loss_fn(y_true, y_pred):
            return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) + sum(model.losses)

        model.compile(optimizer=tf.optimizers.get({'class_name': training_dict['optimizer'],
                                                   'config': {'learning_rate': training_dict['learning_rate']}}),
                      loss=loss_fn,
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model

    def model_fn(model_class=tf.keras.Sequential, train_size=None):
        model = compile_model(create_model(model_class, train_size))
        if training_dict['method'] == 'fedavg':
            return tff.learning.from_compiled_keras_model(model, sample_batch)
        return model

    return model_fn


def run_simulation(model_fn, federated_train_data, federated_test_data, train_size, test_size, model, training, logdir):
    if training['method'] == 'virtual':
        virtual_process = VirtualFedProcess(model_fn, model['num_clients'], damping_factor=training['damping_factor'],
                                                                            fed_avg_init=training['fed_avg_init'])
        virtual_process.fit(federated_train_data, training['num_rounds'], training['clients_per_round'],
                            training['epochs_per_round'], train_size=train_size, test_size=test_size,
                            federated_test_data=federated_test_data,
                            tensorboard_updates=training['tensorboard_updates'],
                            callbacks=training['callbacks'], logdir=logdir)
    elif training['method'] == 'fedavg':
        train_log_dir = logdir + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(logdir)

        tff.framework.set_default_executor(tff.framework.create_local_executor())
        iterative_process = tff.learning.build_federated_averaging_process(model_fn)
        evaluation = tff.learning.build_federated_evaluation(model_fn)
        state = iterative_process.initialize()

        for round_num in range(training['num_rounds']):
            state, metrics = iterative_process.next(state, [federated_train_data[indx] for indx in
                                                            random.sample(range(model['num_clients']),
                                                                          training['clients_per_round'])])
            test_metrics = evaluation(state.model, federated_test_data)
            print('round {:2d}, metrics_train={}, metrics_test={}'.format(round_num, metrics, test_metrics))
            if round_num % training['tensorboard_updates'] == 0:
                with train_summary_writer.as_default():
                    for name, value in metrics._asdict().items():
                        tf.summary.scalar(name, value, step=round_num)
                with test_summary_writer.as_default():
                    for name, value in test_metrics._asdict().items():
                        tf.summary.scalar(name, value, step=round_num)

    elif training['method'] == 'fedprox':
        fed_prox_process = FedProx(model_fn, model['num_clients'])
        fed_prox_process.fit(federated_train_data, training['num_rounds'], training['clients_per_round'],
                            training['epochs_per_round'], train_size=train_size, test_size=test_size,
                            federated_test_data=federated_test_data,
                            tensorboard_updates=training['tensorboard_updates'],
                            logdir=logdir)
