import os
import random
import logging

import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, RNN, Dense, Embedding, LSTMCell
import gc

from source.virtual_process import VirtualFedProcess
from source.fed_prox import FedProx
from source.gate_layer import Gate
from source.utils import FlattenedCategoricalAccuracy
from source.federated_devices import _Server
from source.centered_layers import (DenseCentered, CenteredL2Regularizer,
                                    EmbeddingCentered, LSTMCellCentered,
                                    RNNCentered, Conv2DCentered)
from source.natural_raparametrization_layer import RNNVarReparametrized
from source.natural_raparametrization_layer import Conv1DVirtualNatural
from source.natural_raparametrization_layer import Conv2DVirtualNatural
from source.natural_raparametrization_layer import DenseReparametrizationNaturalShared, \
                                                   DenseLocalReparametrizationNaturalShared,\
                                                   DenseSharedNatural, \
                                                   natural_mean_field_normal_fn, \
                                                   natural_tensor_multivariate_normal_fn, \
                                                   natural_initializer_fn, \
                                                   NaturalGaussianEmbedding, LSTMCellVariationalNatural
from source.tfp_utils import precision_from_untransformed_scale
from source.constants import ROOT_LOGGER_STR
from tensorflow_probability.python.layers import DenseReparameterization
from source.learning_rate_multipliers_opt import LR_SGD
from source.federated_devices import _Server

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


dir_path = os.path.dirname(os.path.realpath(__file__))


def get_compiled_model_fn_from_dict(dict_conf, sample_batch):
    def create_seq_model(model_class=tf.keras.Sequential, train_size=None,
                         client_weight=None):
        # Make sure layer parameters are a list
        if not isinstance(dict_conf['layers'], list):
            dict_conf['layers'] = [dict_conf['layers']]

        layers = []
        for layer_params in dict_conf['layers']:
            layer_params = dict(layer_params)
            layer_class = globals()[layer_params['name']]
            layer_params.pop('name')
            layer_params.pop('scale_init', None)

            def kernel_reg_fn():
                return CenteredL2Regularizer(dict_conf['l2_reg'])

            if not train_size:
                train_size = 1.

            k_w = float(train_size)
            if issubclass(model_class, _Server):
                k_w = 1

            kernel_divergence_fn = (lambda q, p, ignore:
                                    dict_conf['kl_weight']
                                    * kl_lib.kl_divergence(q, p) / k_w)
            reccurrent_divergence_fn = (lambda q, p, ignore:
                                        dict_conf['kl_weight']
                                        * kl_lib.kl_divergence(q, p) / k_w)

            if ('scale_init' in dict_conf
                    and (issubclass(layer_class, DenseSharedNatural)
                         or layer_class == Conv2DVirtualNatural
                         or layer_class == NaturalGaussianEmbedding
                         or layer_class == LSTMCellVariationalNatural)):
                scale_init = dict_conf['scale_init']
                untransformed_scale = scale_init[0]
                if scale_init[0] == 'auto':
                    untransformed_scale = \
                        precision_from_untransformed_scale.inverse(
                            tf.constant(train_size, dtype=tf.float32))
                layer_params['untransformed_scale_initializer'] = \
                    tf.random_normal_initializer(mean=untransformed_scale,
                                                 stddev=scale_init[1])

            if ('loc_init' in dict_conf
                    and (issubclass(layer_class, DenseSharedNatural)
                         or layer_class == Conv2DVirtualNatural
                         or layer_class == NaturalGaussianEmbedding)):
                loc_init = dict_conf['loc_init']
                layer_params['loc_initializer'] = \
                    tf.random_normal_initializer(mean=loc_init[0],
                                                 stddev=loc_init[1])

            if layer_class == DenseReparameterization:
                layer_params['kernel_divergence_fn'] = kernel_divergence_fn
            if issubclass(layer_class, DenseSharedNatural):
                layer_params['kernel_divergence_fn'] = kernel_divergence_fn
                layer_params['client_weight'] = client_weight
                layer_params['delta_percentile'] = dict_conf.get('delta_percentile', None)
            if layer_class == Conv2DVirtualNatural:
                layer_params['kernel_divergence_fn'] = kernel_divergence_fn
                layer_params['client_weight'] = client_weight
            if layer_class == DenseCentered:
                layer_params['kernel_regularizer'] = kernel_reg_fn
                layer_params['bias_regularizer'] = kernel_reg_fn
            if layer_class == EmbeddingCentered:
                layer_params['embeddings_regularizer'] = kernel_reg_fn
                layer_params['batch_input_shape'] = [dict_conf['batch_size'],
                                                     dict_conf['seq_length']]
                layer_params['mask_zero'] = True
            if layer_class == Conv2DCentered:
                layer_params['kernel_regularizer'] = \
                    lambda: CenteredL2Regularizer(dict_conf['l2_reg'])
                layer_params['bias_regularizer'] = \
                    lambda: CenteredL2Regularizer(dict_conf['l2_reg'])
            if layer_class == NaturalGaussianEmbedding:
                layer_params['embedding_divergence_fn'] = kernel_divergence_fn
                layer_params['batch_input_shape'] = [dict_conf['batch_size'],
                                                     dict_conf['seq_length']]
                layer_params['mask_zero'] = True
                if layer_class == NaturalGaussianEmbedding:
                    layer_params['client_weight'] = client_weight

            if layer_class == LSTMCellCentered:
                cell_params = dict(layer_params)
                cell_params['kernel_regularizer'] = kernel_reg_fn
                cell_params['recurrent_regularizer'] = kernel_reg_fn
                cell_params['bias_regularizer'] = kernel_reg_fn
                cell = layer_class(**cell_params)
                layer_params = {'cell': cell,
                               'return_sequences': True,
                               'stateful': True}
                layer_class = RNNCentered
            if layer_class == LSTMCellVariationalNatural:
                cell_params = dict(layer_params)
                if layer_class == LSTMCellVariationalNatural:
                    cell_params['client_weight'] = client_weight
                cell_params['kernel_divergence_fn'] = kernel_divergence_fn
                cell_params['recurrent_kernel_divergence_fn'] = \
                    reccurrent_divergence_fn
                cell = layer_class(**cell_params)

                layer_params = {'cell': cell,
                                'return_sequences': True,
                                'stateful': True}
                layer_class = RNNVarReparametrized

            layer_params.pop('name', None)
            layers.append(layer_class(**layer_params))
        return model_class(layers)

    def create_model_hierarchical(model_class=tf.keras.Model, train_size=None,
                                  client_weight=None):
        if 'architecture' in dict_conf and dict_conf['architecture'] == 'rnn':
            b_shape = (dict_conf['batch_size'], dict_conf['seq_length'])
            in_layer = tf.keras.layers.Input(batch_input_shape=b_shape)
        else:
            in_key = ('input_dim' if 'input_dim' in dict_conf['layers'][0]
                      else 'input_shape')
            input_dim = dict_conf['layers'][0][in_key]
            in_layer = tf.keras.layers.Input(shape=input_dim)

        client_path = in_layer
        server_path = in_layer

        for layer_params in dict_conf['layers']:
            layer_params = dict(layer_params)
            layer_class = globals()[layer_params['name']]
            layer_params.pop('name')

            k_w = float(train_size)
            if issubclass(model_class, _Server):
                k_w = 1

            server_divergence_fn = (lambda q, p, ignore:
                                    dict_conf['kl_weight']
                                    * kl_lib.kl_divergence(q, p) / k_w)
            client_divergence_fn = (lambda q, p, ignore:
                                    dict_conf['kl_weight']
                                    * kl_lib.kl_divergence(q, p) / k_w)

            client_posterior_fn = natural_mean_field_normal_fn
            client_prior_fn = natural_tensor_multivariate_normal_fn

            client_reccurrent_divergence_fn = (lambda q, p, ignore:
                                               dict_conf['kl_weight']
                                               * kl_lib.kl_divergence(q, p)
                                               / k_w)
            server_reccurrent_divergence_fn = (lambda q, p, ignore:
                                               dict_conf['kl_weight']
                                               * kl_lib.kl_divergence(q, p)
                                               / k_w)

            if ('scale_init' in dict_conf
                    and (issubclass(layer_class, DenseSharedNatural)
                         or layer_class == Conv2DVirtualNatural)
                         or layer_class == NaturalGaussianEmbedding):
                scale_init = dict_conf['scale_init']
                untransformed_scale = scale_init[0]
                if scale_init[0] == 'auto':
                    untransformed_scale = \
                        precision_from_untransformed_scale.inverse(
                            tf.constant(train_size, dtype=tf.float32))
                layer_params['untransformed_scale_initializer'] = \
                    tf.random_normal_initializer(mean=untransformed_scale,
                                                 stddev=scale_init[1])
            if ('loc_init' in dict_conf
                    and (issubclass(layer_class, DenseSharedNatural)
                         or layer_class == Conv2DVirtualNatural
                         or layer_class == NaturalGaussianEmbedding)):
                loc_init = dict_conf['loc_init']
                layer_params['loc_initializer'] = \
                    tf.random_normal_initializer(mean=loc_init[0],
                                                 stddev=loc_init[1])

            if issubclass(layer_class, DenseSharedNatural):
                server_params = dict(layer_params)
                server_params['kernel_divergence_fn'] = server_divergence_fn
                if issubclass(layer_class, DenseSharedNatural):
                    server_params['client_weight'] = client_weight
                client_params = dict(layer_params)
                client_params['kernel_divergence_fn'] = client_divergence_fn
                server_params['activation'] = 'linear'
                if issubclass(layer_class, DenseSharedNatural):
                    natural_initializer = natural_initializer_fn(
                        untransformed_scale_initializer=tf.random_normal_initializer(mean=-5, stddev=0.1))
                    client_params['kernel_posterior_fn'] = client_posterior_fn(natural_initializer)
                    client_params['kernel_prior_fn'] = client_prior_fn()

                client_params.pop('untransformed_scale_initializer', None)
                client_params.pop('loc_initializer', None)
                print('client par:', client_params)
                client_path = tfp.layers.DenseReparameterization(
                    **client_params)(client_path)
                print('server par:', server_params, 'layer_class', layer_class)
                server_path = layer_class(**server_params)(server_path)
                gate_initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=0.1)
                if issubclass(model_class, _Server):
                    print('use zero initializaer')
                    gate_initializer = tf.keras.initializers.Constant(0.)
                server_path = tf.keras.layers.Activation(
                     activation=layer_params['activation'])(
                     tf.keras.layers.Add()([server_path, Gate(gate_initializer)(client_path)]))

            elif issubclass(layer_class, Conv2DVirtualNatural):
                client_params = dict(layer_params)
                server_params = dict(layer_params)

                if issubclass(layer_class, Conv2DVirtualNatural):
                    natural_initializer = natural_initializer_fn(
                        untransformed_scale_initializer=
                        layer_params['untransformed_scale_initializer'])
                    client_params['kernel_posterior_fn'] = client_posterior_fn(natural_initializer)
                    client_params['kernel_prior_fn'] = client_prior_fn()
                    server_params['client_weight'] = client_weight

                client_params['kernel_divergence_fn'] = client_divergence_fn
                client_params['activation'] = 'linear'
                client_params.pop('untransformed_scale_initializer', None)
                client_params.pop('loc_initializer', None)
                client_path = tfp.layers.Convolution2DReparameterization(
                    **client_params)(client_path)

                server_params['kernel_divergence_fn'] = server_divergence_fn
                server_path = layer_class(**server_params)(server_path)
                gate_initializer = tf.keras.initializers.RandomUniform(
                    minval=0, maxval=0.1)
                if issubclass(model_class, _Server):
                    print('use zero initializaer')
                    gate_initializer = tf.keras.initializers.Constant(0.)
                server_path = tf.keras.layers.Activation(
                    activation=layer_params['activation'])(
                    tf.keras.layers.Add()(
                        [server_path, Gate(gate_initializer)(client_path)]))
            else:
                client_path = layer_class(**layer_params)(client_path)
                server_path = layer_class(**layer_params)(server_path)

        return model_class(inputs=in_layer, outputs=server_path)

    def compile_model(model, client_weight=None):
        if not client_weight:
            client_weight = 1.

        def loss_fn(y_true, y_pred):
            return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) + sum(model.losses)

        metric = tf.keras.metrics.SparseCategoricalAccuracy()
        if 'architecture' in dict_conf:
            if dict_conf['architecture'] == 'rnn':
                metric = FlattenedCategoricalAccuracy(vocab_size=dict_conf['vocab_size'])

        if "decay_rate" in dict_conf:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                dict_conf['learning_rate'],
                decay_steps=dict_conf['decay_steps'],
                decay_rate=dict_conf['decay_rate'],
                staircase=True)
        else:
            lr_schedule = dict_conf['learning_rate']

        if "momentum" in dict_conf:  # Case of SGD
            optimizer = tf.optimizers.get(
                {'class_name': dict_conf['optimizer'],
                 'config': {'learning_rate': lr_schedule,
                            'momentum': dict_conf['momentum'],
                            'nesterov': dict_conf.get('nesterov', False)}})
        elif "beta" in dict_conf:  # Case of Adam
            optimizer = tf.optimizers.get(
                {'class_name': dict_conf['optimizer'],
                 'config': {'learning_rate': lr_schedule,
                            'beta_1': dict_conf['beta'][0],
                            'beta_2': dict_conf['beta'][1],
                            'amsgrad': dict_conf.get('amsgrad', False)}})
        else:
            optimizer = tf.optimizers.get(
                {'class_name': dict_conf['optimizer'],
                 'config': {'learning_rate': lr_schedule}})

        if dict_conf['optimizer'] == 'sgd':
            LR_mult_dict = {}
            for layer in model.layers:
                layer_to_check = layer
                if hasattr(layer, 'cell'):
                    layer_to_check = layer.cell
                if 'natural' in layer_to_check.name:
                    LR_mult_dict[layer.name] = 1 / (lr_schedule * client_weight) * dict_conf['natural_lr']
                elif 'dense_reparameterization' in layer.name:
                    LR_mult_dict[layer.name] = 1 / lr_schedule * dict_conf['natural_lr']

            optimizer = LR_SGD(lr=lr_schedule, multipliers=LR_mult_dict)

        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=[metric],
                      experimental_run_tf_function=False)
        return model

    def model_fn(model_class=tf.keras.Sequential, train_size=None, client_weight=None):
        create = create_seq_model
        if 'hierarchical' in dict_conf and dict_conf['hierarchical']:
            create = create_model_hierarchical

        model = compile_model(create(model_class, train_size, client_weight), client_weight)
        if dict_conf['method'] == 'fedavg':
            return tff.learning.from_compiled_keras_model(model, sample_batch)
        return model

    return model_fn


def run_simulation(model_fn, federated_train_data, federated_test_data,
                   train_size, test_size, cfgs, logdir):
    if cfgs['method'] == 'virtual':
        virtual_process = VirtualFedProcess(model_fn, cfgs['num_clients'],
                                            damping_factor=cfgs['damping_factor'],
                                            fed_avg_init=cfgs['fed_avg_init'])
        virtual_process.fit(federated_train_data, cfgs['num_rounds'],
                            cfgs['clients_per_round'],
                            cfgs['epochs_per_round'],
                            train_size=train_size, test_size=test_size,
                            federated_test_data=federated_test_data,
                            tensorboard_updates=cfgs['tensorboard_updates'],
                            logdir=logdir, hierarchical=cfgs['hierarchical'],
                            verbose=cfgs['verbose'],
                            server_learning_rate=cfgs['server_learning_rate'],
                            MTL=True)
        tf.keras.backend.clear_session()
        del virtual_process
        gc.collect()
    elif cfgs['method'] == 'fedavg':
        train_log_dir = logdir / 'train'
        train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
        test_summary_writer = tf.summary.create_file_writer(str(logdir))

        tff.framework.set_default_executor(tff.framework.create_local_executor())
        iterative_process = tff.learning.build_federated_averaging_process(model_fn)
        evaluation = tff.learning.build_federated_evaluation(model_fn)
        state = iterative_process.initialize()

        for round_num in range(cfgs['num_rounds']):
            state, metrics = iterative_process.next(
                state,
                [federated_train_data[indx]
                 for indx in random.sample(range(cfgs['num_clients']),
                                           cfgs['clients_per_round'])])
            test_metrics = evaluation(state.model, federated_test_data)
            logger.info(f'round {round_num:2d}, '
                        f'metrics_train={metrics}, '
                        f'metrics_test={test_metrics}')
            if round_num % cfgs['tensorboard_updates'] == 0:
                with train_summary_writer.as_default():
                    for name, value in metrics._asdict().items():
                        tf.summary.scalar(name, value, step=round_num)
                with test_summary_writer.as_default():
                    for name, value in test_metrics._asdict().items():
                        tf.summary.scalar(name, value, step=round_num)

    elif cfgs['method'] == 'fedprox':
        fed_prox_process = FedProx(model_fn, cfgs['num_clients'])
        fed_prox_process.fit(federated_train_data, cfgs['num_rounds'],
                             cfgs['clients_per_round'],
                             cfgs['epochs_per_round'],
                             train_size=train_size, test_size=test_size,
                             federated_test_data=federated_test_data,
                             tensorboard_updates=cfgs['tensorboard_updates'],
                             logdir=logdir,
                             verbose=cfgs['verbose'],
                             server_learning_rate=cfgs['server_learning_rate'],
                             MTL=False)
        tf.keras.backend.clear_session()
        del fed_prox_process
        gc.collect()
