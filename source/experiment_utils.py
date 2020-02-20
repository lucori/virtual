import os
import tensorflow as tf
from dense_reparametrization_shared import DenseReparametrizationShared
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
import tensorflow_federated as tff
from virtual_process import VirtualFedProcess
import random
from fed_prox import FedProx
from centered_layers import DenseCentered, CenteredL2Regularizer
from gate_layer import Gate
from tensorflow.keras.layers import Dense

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_compiled_model_fn_from_dict(dict_conf, sample_batch):
    layer = globals()[dict_conf['layer']]

    def create_model(model_class=tf.keras.Sequential, train_size=None):
        args = {}
        if layer == DenseReparametrizationShared:
            kernel_divergence_fn = (lambda q, p, ignore:  dict_conf['kl_weight'] * kl_lib.kl_divergence(q, p) / float(train_size))
            args['kernel_divergence_fn'] = kernel_divergence_fn
            args['num_clients'] = dict_conf['num_clients']
            args['prior_scale'] = dict_conf['prior_scale']
        if layer == DenseCentered:
            args['kernel_regularizer'] = lambda: CenteredL2Regularizer(dict_conf['l2_reg'])
            args['bias_regularizer'] = lambda: CenteredL2Regularizer(dict_conf['l2_reg'])
        layers = []
        for i, (l_u, act) in enumerate(zip(dict_conf['layer_units'], dict_conf['activations'])):
            if i == 0:
                args['input_shape'] = dict_conf['input_shape']
            args['activation'] = act
            layers.append(layer(l_u, **args))
            args.pop('input_shape', None)

        return model_class(layers)

    def create_model_hierarchical(model_class=tf.keras.Model, train_size=None):
        args_client = {}
        args_server = {}
        if layer == DenseReparametrizationShared:
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

        model.compile(optimizer=tf.optimizers.get({'class_name': dict_conf['optimizer'],
                                                   'config': {'learning_rate': dict_conf['learning_rate']}}),
                      loss=loss_fn,
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model

    def model_fn(model_class=tf.keras.Sequential, train_size=None):
        create = create_model
        if 'hierarchical' in dict_conf:
            if dict_conf['hierarchical']:
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
