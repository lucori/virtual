import random
import logging
from pathlib import Path
import tensorflow as tf
import numpy as np
from source.federated_devices import (ClientVirtualSequential,
                                      ClientVirtualModel,
                                      ServerSequential,
                                      ServerModel)
from source.fed_process import FedProcess
from source.utils import avg_dict, avg_dict_eval
from source.constants import ROOT_LOGGER_STR
from operator import itemgetter
from source.utils import CustomTensorboard

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class VirtualFedProcess(FedProcess):

    def __init__(self, model_fn, num_clients, damping_factor=1,
                 fed_avg_init=True):
        super(VirtualFedProcess, self).__init__(model_fn, num_clients)
        self.damping_factor = damping_factor
        self.fed_avg_init = fed_avg_init

    def build(self, cards_train, hierarchical):
        if hierarchical:
            client_model_class = ClientVirtualModel
            server_model_class = ServerModel
        else:
            client_model_class = ClientVirtualSequential
            server_model_class = ServerSequential
        for indx in self.clients_indx:
            model = self.model_fn(client_model_class, cards_train[indx], cards_train[indx]/sum(cards_train))
            self.clients.append(model)
        self.server = self.model_fn(server_model_class, sum(cards_train), 1.)

    def fit(self, federated_train_data, num_rounds, clients_per_round,
            epochs_per_round, federated_test_data=None,
            tensorboard_updates=1, logdir=Path(), callbacks=None,
            train_size=None, test_size=None, hierarchical=False):

        train_log_dir = logdir / 'train'
        server_log_dir = logdir / 'server'
        all_client_log_dir = logdir / 'client_all'
        selected_client_log_dir = logdir / 'client_selected'

        self.train_summary_writer = \
            tf.summary.create_file_writer(str(train_log_dir))
        self.server_summary_writer = \
            tf.summary.create_file_writer(str(server_log_dir))
        self.selected_client_summary_writer = \
            tf.summary.create_file_writer(str(selected_client_log_dir))
        self.all_client_summary_writer = \
            tf.summary.create_file_writer(str(all_client_log_dir))

        self.build(train_size, hierarchical)

        history_test = [None] * len(self.clients)
        max_accuracy = -1.0
        max_acc_round = None
        updated_clients = [False] * len(self.clients)

        deltas = [client.compute_delta() for client in self.clients]
        aggregated_deltas = self.aggregate_deltas_multi_layer(
            deltas, [size/sum(train_size) for size in train_size])
        self.server.apply_delta(aggregated_deltas)

        server_test_accs = np.zeros(num_rounds)
        all_client_test_accs = np.zeros(num_rounds)
        selected_client_test_accs = np.zeros(num_rounds)
        training_accs = np.zeros(num_rounds)
        server_test_losses = np.zeros(num_rounds)
        all_client_test_losses = np.zeros(num_rounds)
        selected_client_test_losses = np.zeros(num_rounds)
        training_losses = np.zeros(num_rounds)
        for round_i in range(num_rounds):
            clients_sampled = random.sample(self.clients_indx,
                                            clients_per_round)
            deltas = []
            history_train = []
            for indx in clients_sampled:
                self.clients[indx].receive_and_save_weights(self.server)
                self.clients[indx].renew_center()
                if self.fed_avg_init or not self.clients[indx].s_i_to_update:
                    print('initialize posterior with server')
                    #self.clients[indx].initialize_kernel_posterior()

                history_single = self.clients[indx].fit(
                    federated_train_data[indx],
                    verbose=1,
                    validation_data=federated_test_data[indx],
                    epochs=epochs_per_round,
                    callbacks=#callbacks
                    [CustomTensorboard(log_dir='logs/femnist_100_10/test/',
                                       histogram_freq=1, profile_batch=0)]
                )

                self.clients[indx].apply_damping(self.damping_factor)

                updated_clients[indx] = True
                self.clients[indx].s_i_to_update = True

                delta = self.clients[indx].compute_delta()
                deltas.append(delta)

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
                                tf.summary.histogram(
                                    layer.name + '/server_gamma',
                                    value[..., 0], step=round_i)
                                tf.summary.histogram(
                                    layer.name + '/server_prec',
                                    value[..., 1], step=round_i)

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
            aggregated_deltas = self.aggregate_deltas_multi_layer(
                deltas,
                #[1. for _ in self.clients]
                [train_size[client] / sum(train_size) for client in clients_sampled]
                #[train_size[client] / sum(train_size_sampled) for client in clients_sampled]
                )
            self.server.apply_delta(aggregated_deltas)

            server_test = [self.server.evaluate(test_data, verbose=0)
                           for test_data in federated_test_data]
            all_client_test = [self.clients[indx].evaluate(test_data, verbose=0)
                               for indx, test_data in enumerate(federated_test_data)]

            avg_train = avg_dict(history_train,
                                 [train_size[client]
                                  for client in clients_sampled])
            selected_client_test = avg_dict(history_test, test_size)
            server_avg_test = avg_dict_eval(
                server_test, [size / sum(test_size) for size in test_size])
            all_client_avg_test = avg_dict_eval(
                all_client_test, [size / sum(test_size) for size in test_size])

            if server_avg_test[1] > max_accuracy:
                max_accuracy = server_avg_test[1]
                max_acc_round = round_i
            server_test_accs[round_i] = server_avg_test[1]
            all_client_test_accs[round_i] = all_client_avg_test[1]
            training_accs[round_i] = avg_train['sparse_categorical_accuracy']
            selected_client_test_accs[round_i] = selected_client_test['sparse_categorical_accuracy']

            server_test_losses[round_i] = server_avg_test[0]
            all_client_test_losses[round_i] = all_client_avg_test[0]
            selected_client_test_losses[round_i] = selected_client_test['loss']
            training_losses[round_i] = avg_train['loss']
            logger.debug(f"round: {round_i}, "
                         f"avg_train: {avg_train}, "
                         f"selected_client_test: {selected_client_test}, "
                         f"server_avg_test on whole test data: {server_avg_test} "
                         f"all clients avg test: {all_client_avg_test}, "
                         f"max accuracy so far: {max_accuracy} reached at "
                         f"round {max_acc_round}")
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
                with self.server_summary_writer.as_default():
                    tf.summary.scalar('max_sparse_categorical_accuracy',
                                      max_accuracy, step=round_i)

            # Do this at every round to make sure to keep the data even if
            # the training is interrupted
            np.save(logdir.parent / 'server_accs.npy', server_test_accs)
            np.save(logdir.parent / 'all_client_accs.npy', all_client_test_accs)
            np.save(logdir.parent / 'training_accs.npy', training_accs)
            np.save(logdir.parent / 'selected_client_accs.npy', selected_client_test_accs)

            np.save(logdir.parent / 'server_losses.npy', server_test_losses)
            np.save(logdir.parent / 'all_client_losses.npy', all_client_test_losses)
            np.save(logdir.parent / 'training_losses.npy', training_losses)
            np.save(logdir.parent / 'selected_client_losses.npy', selected_client_test_losses)

        for i, client in enumerate(self.clients):
            client.save_weights(str(logdir / f'weights_{i}.h5'))
