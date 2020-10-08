import tensorflow as tf
from source.tfp_utils import loc_prod_from_locprec
eps = 1/tf.float32.max
import random
import logging
from pathlib import Path
import tensorflow as tf
import numpy as np
from source.utils import avg_dict, avg_dict_eval
from source.constants import ROOT_LOGGER_STR
from operator import itemgetter
from source.utils import CustomTensorboard

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class FedProcess:

    def __init__(self, model_fn, num_clients):
        self.model_fn = model_fn
        self.num_clients = num_clients
        self.clients_indx = range(self.num_clients)
        self.clients = []
        self.server = None

        self.train_summary_writer = None
        self.test_summary_writer = None
        self.valid_summary_writer = None

    def build(self, *args, **kwargs):
        pass

    def aggregate_deltas_multi_layer(self, deltas, client_weight=None):
        aggregated_deltas = []
        deltas = list(map(list, zip(*deltas)))
        for delta_layer in deltas:
            aggregated_deltas.append(
                self.aggregate_deltas_single_layer(delta_layer, client_weight))
        return aggregated_deltas

    def aggregate_deltas_single_layer(self, deltas, client_weight=None):
        for i, delta_client in enumerate(deltas):
            for key, el in delta_client.items():
                if isinstance(el, tuple):
                    (loc, prec) = el
                    if client_weight:
                        prec = prec*client_weight[i]*self.num_clients
                    loc = tf.math.multiply(loc, prec)
                    delta_client[key] = (loc, prec)
                else:
                    if client_weight:
                        delta_client[key] = (el*client_weight[i], )
                    else:
                        delta_client[key] = (el/self.num_clients, )

        deltas = {key: [dic[key] for dic in deltas] for key in deltas[0]}
        for key, lst in deltas.items():
            lst = zip(*lst)
            sum_el = []
            for i, el in enumerate(lst):
                add = tf.math.add_n(el)
                sum_el.append(add)

            if len(sum_el) == 2:
                loc = loc_prod_from_locprec(*sum_el)
                deltas[key] = (loc, sum_el[1])
            else:
                deltas[key] = sum_el[0]
        return deltas

    def fit(self,
            federated_train_data,
            num_rounds,
            clients_per_round,
            epochs_per_round,
            federated_test_data=None,
            tensorboard_updates=1,
            logdir=Path(),
            callbacks=None,
            train_size=None,
            test_size=None,
            hierarchical=False,
            server_learning_rate=1.,
            verbose=0,
            MTL=False):

        train_log_dir = logdir / 'train'
        server_log_dir = logdir / 'server'
        selected_client_log_dir = logdir / 'client_selected'
        all_client_log_dir = logdir / 'client_all'
        self.all_client_summary_writer = \
            tf.summary.create_file_writer(str(all_client_log_dir))
        self.train_summary_writer = \
            tf.summary.create_file_writer(str(train_log_dir))
        self.server_summary_writer = \
            tf.summary.create_file_writer(str(server_log_dir))
        self.selected_client_summary_writer = \
            tf.summary.create_file_writer(str(selected_client_log_dir))

        if MTL:
            self.build(train_size, hierarchical)
            deltas = [client.compute_delta() for client in self.clients]
            aggregated_deltas = self.aggregate_deltas_multi_layer(
                deltas, [size / sum(train_size) for size in train_size])
            self.server.apply_delta(aggregated_deltas)
        else:
            self.build()

        history_test = [None] * len(self.clients)
        max_accuracy = -1.0
        max_acc_round = None
        server_test_accs = np.zeros(num_rounds)
        all_client_test_accs = np.zeros(num_rounds)
        selected_client_test_accs = np.zeros(num_rounds)
        training_accs = np.zeros(num_rounds)
        server_test_losses = np.zeros(num_rounds)
        all_client_test_losses = np.zeros(num_rounds)
        selected_client_test_losses = np.zeros(num_rounds)
        training_losses = np.zeros(num_rounds)
        overall_tensorboard = CustomTensorboard(log_dir=str(train_log_dir),
                                                histogram_freq=max(0, verbose - 2),
                                                profile_batch=max(0, verbose - 2))
        if verbose >= 2:
            if callbacks:
                callbacks.append(overall_tensorboard)
            else:
                callbacks = [overall_tensorboard]

        for round_i in range(num_rounds):
            clients_sampled = random.sample(self.clients_indx,
                                            clients_per_round)
            deltas = []
            history_train = []
            for indx in clients_sampled:
                self.clients[indx].receive_and_save_weights(self.server)
                self.clients[indx].renew_center(round_i > 0)

                if MTL:
                    if self.fed_avg_init == 2 or (
                            self.fed_avg_init
                            and not self.clients[indx].s_i_to_update):
                        print('initialize posterior with server')
                        self.clients[indx].initialize_kernel_posterior()

                history_single = self.clients[indx].fit(
                    federated_train_data[indx],
                    verbose=0,
                    validation_data=federated_test_data[indx],
                    epochs=epochs_per_round,
                    callbacks=callbacks
                )

                if MTL:
                    self.clients[indx].apply_damping(self.damping_factor)

                delta = self.clients[indx].compute_delta()
                deltas.append(delta)

                if verbose >= 1:
                    with self.train_summary_writer.as_default():
                        for layer in self.clients[indx].layers:
                            for weight in layer.trainable_weights:
                                if 'natural' in weight.name:
                                    tf.summary.histogram(
                                        layer.name + '/gamma',
                                        weight[..., 0], step=round_i)
                                    tf.summary.histogram(
                                        layer.name + '/prec',
                                        weight[..., 1], step=round_i)
                                else:
                                    tf.summary.histogram(
                                        layer.name, weight, step=round_i)
                            if hasattr(layer, 'kernel_posterior'):
                                tf.summary.histogram(
                                    layer.name + '/gamma_reparametrized',
                                    layer.kernel_posterior.distribution.gamma,
                                    step=round_i)
                                tf.summary.histogram(
                                    layer.name + '/prec_reparametrized',
                                    layer.kernel_posterior.distribution.prec,
                                    step=round_i)
                        for layer in self.server.layers:
                            if hasattr(layer, 'server_variable_dict'):
                                for key, value in layer.server_variable_dict.items():
                                    if 'natural' in layer.name:
                                        tf.summary.histogram(
                                            layer.name + '/server_gamma',
                                            value[..., 0], step=round_i)
                                        tf.summary.histogram(
                                            layer.name + '/server_prec',
                                            value[..., 1], step=round_i)
                                    else:
                                        tf.summary.histogram(layer.name, value, step=round_i)

                history_train.append({key: history_single.history[key]
                                      for key in history_single.history.keys()
                                      if 'val' not in key})
                history_test[indx] = \
                    {key.replace('val_', ''): history_single.history[key]
                     for key in history_single.history.keys()
                     if 'val' in key}

            train_size_sampled = itemgetter(*clients_sampled)(train_size)
            if clients_per_round == 1:
                train_size_sampled = [train_size_sampled]

            if MTL:
                client_weights = [server_learning_rate * train_size[client] / sum(train_size)
                                  for client in clients_sampled]
            else:
                client_weights = [server_learning_rate * train_size[client] / sum(train_size_sampled)
                                  for client in
                                  clients_sampled]

            aggregated_deltas = self.aggregate_deltas_multi_layer(deltas, client_weights)
            self.server.apply_delta(aggregated_deltas)

            server_test = [self.server.evaluate(test_data, verbose=0)
                           for test_data in federated_test_data]

            all_client_test = [self.clients[indx].evaluate(test_data, verbose=0)
                               for indx, test_data in enumerate(federated_test_data)]
            all_client_avg_test = avg_dict_eval(
                all_client_test, [size / sum(test_size) for size in test_size])
            all_client_test_accs[round_i] = all_client_avg_test[1]
            all_client_test_losses[round_i] = all_client_avg_test[0]

            avg_train = avg_dict(history_train,
                                 [train_size[client]
                                  for client in clients_sampled])
            selected_client_test = avg_dict(history_test, test_size)
            server_avg_test = avg_dict_eval(
                server_test, [size / sum(test_size) for size in test_size])

            if server_avg_test[1] > max_accuracy:
                max_accuracy = server_avg_test[1]
                max_acc_round = round_i

            server_test_accs[round_i] = server_avg_test[1]
            training_accs[round_i] = avg_train['sparse_categorical_accuracy']
            selected_client_test_accs[round_i] = selected_client_test['sparse_categorical_accuracy']

            server_test_losses[round_i] = server_avg_test[0]
            selected_client_test_losses[round_i] = selected_client_test['loss']
            training_losses[round_i] = avg_train['loss']

            debug_string = (f"round: {round_i}, "
                            f"avg_train: {avg_train}, "
                            f"selected_client_test: {selected_client_test}, "
                            f"server_avg_test on whole test data: {server_avg_test} "
                            f"max accuracy so far: {max_accuracy} reached at "
                            f"round {max_acc_round} "
                            f"all clients avg test: {all_client_avg_test}")
            logger.debug(debug_string)

            if round_i % tensorboard_updates == 0:
                for i, key in enumerate(avg_train.keys()):
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar(key, avg_train[key], step=round_i)
                    with self.server_summary_writer.as_default():
                        tf.summary.scalar(key, server_avg_test[i], step=round_i)
                    with self.selected_client_summary_writer.as_default():
                        tf.summary.scalar(key, selected_client_test[key], step=round_i)
                    with self.all_client_summary_writer.as_default():
                        tf.summary.scalar(key, all_client_avg_test[i], step=round_i)
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('max_sparse_categorical_accuracy',
                                      max_accuracy, step=round_i)

            # Do this at every round to make sure to keep the data even if
            # the training is interrupted
            np.save(logdir.parent / 'server_accs.npy', server_test_accs)
            np.save(logdir.parent / 'training_accs.npy', training_accs)
            np.save(logdir.parent / 'selected_client_accs.npy', selected_client_test_accs)
            np.save(logdir.parent / 'server_losses.npy', server_test_losses)
            np.save(logdir.parent / 'training_losses.npy', training_losses)
            np.save(logdir.parent / 'selected_client_losses.npy', selected_client_test_losses)
            np.save(logdir.parent / 'all_client_accs.npy', all_client_test_accs)
            np.save(logdir.parent / 'all_client_losses.npy', all_client_test_losses)

        for i, client in enumerate(self.clients):
            client.save_weights(str(logdir / f'weights_{i}.h5'))

