import os
import random
import logging

import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, RNN
import gc

from source.virtual_process import VirtualFedProcess
from source.fed_prox import FedProx
from source.gate_layer import Gate
from source.utils import FlattenedCategoricalAccuracy
from source.federated_devices import _Server
from source.centered_layers import (DenseCentered, CenteredL2Regularizer,
                                    EmbeddingCentered, LSTMCellCentered,
                                    RNNCentered, Conv2DCentered)
from source.dense_reparametrization_shared import Conv1DVirtual
from source.dense_reparametrization_shared import Conv2DVirtual
from source.dense_reparametrization_shared import DenseShared
from source.dense_reparametrization_shared import DenseLocalReparametrizationShared
from source.dense_reparametrization_shared import DenseReparametrizationShared
from source.dense_reparametrization_shared import RNNVarReparametrized
from source.dense_reparametrization_shared import RNNReparametrized
from source.dense_reparametrization_shared import GaussianEmbedding
from source.dense_reparametrization_shared import LSTMCellVariational
from source.dense_reparametrization_shared import LSTMCellReparametrization
from source.natural_raparametrization_layer import DenseReparametrizationNaturalShared, \
                                                   DenseLocalReparametrizationNaturalShared,\
                                                   DenseSharedNatural
from source.tfp_utils import precision_from_untransformed_scale
from source.constants import ROOT_LOGGER_STR
from tensorflow_probability.python.layers import DenseReparameterization


logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


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
            layer_params.pop('name')
            layer_params.pop('scale_init', None)

            def kernel_reg_fn():
                return CenteredL2Regularizer(dict_conf['l2_reg'])

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
                    and (issubclass(layer_class, DenseShared) or issubclass(layer_class, DenseSharedNatural)
                         or layer_class == Conv2DVirtual)):
                scale_init = dict_conf['scale_init']
                untransformed_scale = scale_init[0]
                if scale_init[0] == 'auto':
                    untransformed_scale = \
                        precision_from_untransformed_scale.inverse(
                            tf.constant(train_size, dtype=tf.float32))
                layer_params['untransformed_scale_initializer'] = \
                    tf.random_normal_initializer(mean=untransformed_scale,
                                                 stddev=scale_init[1])

            if ('prec_init' in dict_conf
                    and (issubclass(layer_class, DenseShared)
                         or layer_class == Conv2DVirtual)):
                prec_init = dict_conf['prec_init']
                prec_init = prec_init[0]
                if prec_init[0] == 'auto':
                    prec = tf.constant(train_size, dtype=tf.float32)
                layer_params['precision_initializer'] = \
                    tf.random_normal_initializer(mean=prec,
                                                 stddev=prec_init[1])

            if layer_class == DenseReparameterization:
                layer_params['kernel_divergence_fn'] = kernel_divergence_fn
            if issubclass(layer_class, DenseShared) or issubclass(layer_class, DenseSharedNatural):
                layer_params['kernel_divergence_fn'] = kernel_divergence_fn
                layer_params['num_clients'] = dict_conf['num_clients']
                layer_params['prior_scale'] = dict_conf['prior_scale']
            if layer_class == Conv2DVirtual:
                layer_params['kernel_divergence_fn'] = kernel_divergence_fn
                layer_params['num_clients'] = dict_conf['num_clients']
                layer_params['prior_scale'] = dict_conf['prior_scale']
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
            if layer_class == GaussianEmbedding:
                layer_params['embedding_divergence_fn'] = kernel_divergence_fn
                layer_params['num_clients'] = dict_conf['num_clients']
                layer_params['prior_scale'] = dict_conf['prior_scale']
                layer_params['batch_input_shape'] = [dict_conf['batch_size'],
                                                     dict_conf['seq_length']]
                layer_params['mask_zero'] = True
            if layer_class == LSTMCellCentered:
                cell_params = dict(layer_params)
                cell_params['kernel_regularizer'] = kernel_reg_fn
                cell_params['recurrent_regularizer'] = kernel_reg_fn
                cell_params['bias_regularizer'] = kernel_reg_fn
                cell = layer_class(**cell_params)
                layer_params= {'cell': cell,
                               'return_sequences': True,
                               'stateful': True}
                layer_class = RNNCentered
            if layer_class == LSTMCellVariational:
                cell_params = dict(layer_params)
                cell_params['num_clients'] = dict_conf['num_clients']
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

    def create_model_hierarchical(model_class=tf.keras.Model, train_size=None):
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
            client_reccurrent_divergence_fn = (lambda q, p, ignore:
                                               dict_conf['kl_weight']
                                               * kl_lib.kl_divergence(q, p)
                                               / k_w)
            server_reccurrent_divergence_fn = (lambda q, p, ignore:
                                               dict_conf['kl_weight']
                                               * kl_lib.kl_divergence(q, p)
                                               / k_w)

            if ('scale_init' in dict_conf
                    and (issubclass(layer_class, DenseShared)
                         or layer_class == Conv2DVirtual)):
                scale_init = dict_conf['scale_init']
                untransformed_scale = scale_init[0]
                if scale_init[0] == 'auto':
                    untransformed_scale = \
                        precision_from_untransformed_scale.inverse(
                            tf.constant(train_size, dtype=tf.float32))
                layer_params['untransformed_scale_initializer'] = \
                    tf.random_normal_initializer(mean=untransformed_scale,
                                                 stddev=scale_init[1])

            # TODO: Maybe try non-linear activation
            if issubclass(layer_class, DenseShared):
                server_params = dict(layer_params)
                server_params['kernel_divergence_fn'] = server_divergence_fn
                server_params['num_clients'] = dict_conf['num_clients']
                server_params['prior_scale'] = dict_conf['prior_scale']

                client_params = dict(layer_params)
                client_params['kernel_divergence_fn'] = client_divergence_fn
                client_params['activation'] = 'linear'
                client_params.pop('untransformed_scale_initializer', None)

                client_path = tfp.layers.DenseReparameterization(
                    **client_params)(client_path)
                server_path = layer_class(**server_params)(server_path)
                client_path = tf.keras.layers.Activation(
                    activation=layer_params['activation'])(
                    tf.keras.layers.Add()([Gate()(server_path), client_path]))

            elif issubclass(layer_class, Conv2DVirtual):
                client_params = dict(layer_params)
                client_params['kernel_divergence_fn'] = client_divergence_fn
                client_params['activation'] = 'linear'
                client_params.pop('untransformed_scale_initializer', None)
                client_path = tfp.layers.Convolution2DReparameterization(
                    **client_params)(client_path)

                server_params = dict(layer_params)
                server_params['kernel_divergence_fn'] = server_divergence_fn
                server_params['num_clients'] = dict_conf['num_clients']
                server_params['prior_scale'] = dict_conf['prior_scale']
                server_path = layer_class(**server_params)(server_path)

                client_path = tf.keras.layers.Activation(
                    activation=layer_params['activation'])(
                    tf.keras.layers.Add()([Gate()(server_path), client_path]))
            elif issubclass(layer_class, LSTMCellVariational):
                server_params = dict(layer_params)
                server_params['num_clients'] = dict_conf['num_clients']
                server_params['prior_scale'] = dict_conf['prior_scale']
                server_params['kernel_divergence_fn'] = server_divergence_fn
                server_params['recurrent_kernel_divergence_fn'] = \
                    server_reccurrent_divergence_fn
                server_cell = layer_class(**server_params)
                server_params = {'cell': server_cell,
                                 'return_sequences': True,
                                 'stateful': True}
                server_path = RNNVarReparametrized(**server_params)(
                    server_path)

                client_params = dict(layer_params)
                client_params['kernel_divergence_fn'] = client_divergence_fn
                client_params['recurrent_kernel_divergence_fn'] = \
                    client_reccurrent_divergence_fn
                client_cell = LSTMCellReparametrization(**client_params)
                client_params = {'cell': client_cell,
                                 'return_sequences': True,
                                 'stateful': True}
                client_path = RNNReparametrized(**client_params)(client_path)

                client_path = tf.keras.layers.Activation(
                    activation=layer_params['activation'])(
                    tf.keras.layers.Add()([Gate()(server_path), client_path]))
            else:
                client_path = layer_class(**layer_params)(client_path)
                server_path = layer_class(**layer_params)(server_path)

        return model_class(inputs=in_layer, outputs=client_path)

    def compile_model(model):

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

        if "momentum" in dict_conf:
            optimizer = tf.optimizers.get(
                {'class_name': dict_conf['optimizer'],
                 'config': {'learning_rate': lr_schedule,
                            'momentum': dict_conf['momentum']}})
        else:
            optimizer = tf.optimizers.get(
                {'class_name': dict_conf['optimizer'],
                 'config': {'learning_rate': lr_schedule}})

        model.compile(optimizer=optimizer,
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
                            logdir=logdir, hierarchical=cfgs['hierarchical'])
        tf.keras.backend.clear_session()
        del virtual_process
        gc.collect()
    elif cfgs['method'] == 'fedavg':
        train_log_dir = logdir / 'train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(logdir)

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
                             logdir=logdir)
        tf.keras.backend.clear_session()
        del fed_prox_process
        gc.collect()
