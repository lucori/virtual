import os
import random

import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from tensorflow.keras.layers import MaxPooling2D, Flatten
import gc

from source.virtual_process import VirtualFedProcess
from source.fed_prox import FedProx
from source.gate_layer import Gate
from source.utils import FlattenedCategoricalAccuracy
from source.centered_layers import (DenseCentered, CenteredL2Regularizer,
                                    EmbeddingCentered, LSTMCellCentered,
                                    RNNCentered)
from source.dense_reparametrization_shared import Conv1DVirtual
from source.dense_reparametrization_shared import Conv2DVirtual
from source.dense_reparametrization_shared import DenseShared
from source.dense_reparametrization_shared import DenseLocalReparametrizationShared
from source.dense_reparametrization_shared import DenseReparametrizationShared
from source.dense_reparametrization_shared import RNNVarReparametrized
from source.dense_reparametrization_shared import GaussianEmbedding
from source.dense_reparametrization_shared import LSTMCellVariational

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_compiled_model_fn_from_dict(dict_conf, sample_batch):
    def create_seq_model(model_class=tf.keras.Sequential, train_size=None):
        # Make sure layer parameters are a list
        if not isinstance(dict_conf['layers'], list):
            dict_conf['layers'] = [dict_conf['layers']]

        layers = []
        for layer_params in dict_conf['layers']:
            layer_params = dict(layer_params)
            layer_class = globals()[layer_params['name']]
            if issubclass(layer_class, DenseShared):
                kernel_divergence_fn = (lambda q, p, ignore:  dict_conf['kl_weight'] * kl_lib.kl_divergence(q, p) / float(train_size))
                layer_params['kernel_divergence_fn'] = kernel_divergence_fn
                layer_params['num_clients'] = dict_conf['num_clients']
                layer_params['prior_scale'] = dict_conf['prior_scale']
            if layer_class == DenseCentered:
                layer_params['kernel_regularizer'] = lambda: CenteredL2Regularizer(dict_conf['l2_reg'])
                layer_params['bias_regularizer'] = lambda: CenteredL2Regularizer(dict_conf['l2_reg'])
            if layer_class == EmbeddingCentered:
                layer_params['embeddings_regularizer'] = lambda: CenteredL2Regularizer(dict_conf['l2_reg'])
                layer_params['batch_input_shape'] = [dict_conf['batch_size'], dict_conf['seq_length']]
                layer_params['mask_zero'] = True
            if layer_class == GaussianEmbedding:
                embedding_divergence_fn = (
                    lambda q, p, ignore: dict_conf['kl_weight'] * kl_lib.kl_divergence(q, p) / float(train_size))
                layer_params['embedding_divergence_fn'] = embedding_divergence_fn
                layer_params['num_clients'] = dict_conf['num_clients']
                layer_params['prior_scale'] = dict_conf['prior_scale']
                layer_params['batch_input_shape'] = [dict_conf['batch_size'], dict_conf['seq_length']]
                layer_params['mask_zero'] = True
            if layer_class == LSTMCellCentered:
                cell_params = dict(layer_params)
                cell_params['kernel_regularizer'] = lambda: CenteredL2Regularizer(dict_conf['l2_reg'])
                cell_params['recurrent_regularizer'] = lambda: CenteredL2Regularizer(dict_conf['l2_reg'])
                cell_params['bias_regularizer'] = lambda: CenteredL2Regularizer(dict_conf['l2_reg'])
                cell = layer_class(**cell_params)
                layer_params= {'cell': cell,
                               'return_sequences': True,
                               'stateful': True}
                layer_class = RNNCentered
            if layer_class == LSTMCellVariational:
                cell_params = dict(layer_params)
                cell_params['num_clients'] = dict_conf['num_clients']
                cell_params['kernel_divergence_fn'] = (lambda q, p, ignore:
                                                kl_lib.kl_divergence(q, p) / float(train_size))
                cell_params['recurrent_kernel_divergence_fn'] = (lambda q, p, ignore:
                                                          kl_lib.kl_divergence(q, p) / float(train_size))
                cell = layer_class(**cell_params)
                layer_params = {'cell': cell,
                                'return_sequences': True,
                                'stateful': True}
                layer_class = RNNVarReparametrized

            layer_params.pop('name', None)
            layers.append(layer_class(**layer_params))
        return model_class(layers)

    #TODO: add hierarchical RNN
    def create_model_hierarchical(model_class=tf.keras.Model, train_size=None):
        if isinstance(dict_conf['layers'], list):
            layer = [globals()[l['name']] for l in dict_conf['layers']]
        else:
            layer = globals()[dict_conf['layers']]

        args_client = {}
        args_server = {}
        if issubclass(layer, DenseShared):
            kernel_divergence_fn = (
                lambda q, p, ignore: dict_conf['kl_weight'] * kl_lib.kl_divergence(q, p) / float(train_size))
            args_server['kernel_divergence_fn'] = kernel_divergence_fn
            args_server['num_clients'] = dict_conf['num_clients']
            args_server['prior_scale'] = dict_conf['prior_scale']
        input = tf.keras.layers.Input(shape=dict_conf['input_shape'])
        client_path = input
        server_path = input
        for i, (l_u, act) in enumerate(zip(dict_conf['layer_units'], dict_conf['activations'])):
            args_client['activation'] = 'linear'
            client_path = tf.keras.layers.Dense(l_u, **args_client)(client_path)
            args_server['activation'] = act
            server_path = layer(l_u, **args_server)(server_path)
            client_path = tf.keras.layers.Activation(activation=act)(tf.keras.layers.Add()(
                [Gate()(client_path), (server_path)]))

        return model_class(inputs=input, outputs=client_path)

    def compile_model(model):

        def loss_fn(y_true, y_pred):
            return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) + sum(model.losses)

        metric = tf.keras.metrics.SparseCategoricalAccuracy()
        if 'architecture' in dict_conf:
            if dict_conf['architecture'] == 'rnn':
                metric = FlattenedCategoricalAccuracy(vocab_size=dict_conf['vocab_size'])

        model.compile(optimizer=tf.optimizers.get({'class_name': dict_conf['optimizer'],
                                                   'config': {'learning_rate': dict_conf['learning_rate']}}),
                      loss=loss_fn,
                      metrics=[metric],
                      experimental_run_tf_function=False)
        return model

    def model_fn(model_class=tf.keras.Sequential, train_size=None):
        create = create_seq_model
        if 'hierarchical' in dict_conf and dict_conf['hierarchical']:
            create = create_model_hierarchical

        model = compile_model(create(model_class, train_size))
        if dict_conf['method'] == 'fedavg':
            return tff.learning.from_compiled_keras_model(model, sample_batch)
        return model

    return model_fn


def run_simulation(model_fn, federated_train_data, federated_test_data, train_size, test_size, dict_conf, logdir):
    if dict_conf['method'] == 'virtual':
        virtual_process = VirtualFedProcess(model_fn, dict_conf['num_clients'],
                                            damping_factor=dict_conf['damping_factor'],
                                            fed_avg_init=dict_conf['fed_avg_init'])
        virtual_process.fit(federated_train_data, dict_conf['num_rounds'], dict_conf['clients_per_round'],
                            dict_conf['epochs_per_round'], train_size=train_size, test_size=test_size,
                            federated_test_data=federated_test_data,
                            tensorboard_updates=dict_conf['tensorboard_updates'],
                            logdir=logdir, hierarchical=dict_conf['hierarchical'])
        tf.keras.backend.clear_session()
        del virtual_process
        gc.collect()
    elif dict_conf['method'] == 'fedavg':
        train_log_dir = logdir + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(logdir)

        tff.framework.set_default_executor(tff.framework.create_local_executor())
        iterative_process = tff.learning.build_federated_averaging_process(model_fn)
        evaluation = tff.learning.build_federated_evaluation(model_fn)
        state = iterative_process.initialize()

        for round_num in range(dict_conf['num_rounds']):
            state, metrics = iterative_process.next(state, [federated_train_data[indx] for indx in
                                                            random.sample(range(dict_conf['num_clients']),
                                                                          dict_conf['clients_per_round'])])
            test_metrics = evaluation(state.model, federated_test_data)
            print('round {:2d}, metrics_train={}, metrics_test={}'.format(round_num, metrics, test_metrics))
            if round_num % dict_conf['tensorboard_updates'] == 0:
                with train_summary_writer.as_default():
                    for name, value in metrics._asdict().items():
                        tf.summary.scalar(name, value, step=round_num)
                with test_summary_writer.as_default():
                    for name, value in test_metrics._asdict().items():
                        tf.summary.scalar(name, value, step=round_num)

    elif dict_conf['method'] == 'fedprox':
        fed_prox_process = FedProx(model_fn, dict_conf['num_clients'])
        fed_prox_process.fit(federated_train_data, dict_conf['num_rounds'], dict_conf['clients_per_round'],
                            dict_conf['epochs_per_round'], train_size=train_size, test_size=test_size,
                            federated_test_data=federated_test_data,
                            tensorboard_updates=dict_conf['tensorboard_updates'],
                            logdir=logdir)
        tf.keras.backend.clear_session()
        del fed_prox_process
        gc.collect()
