import random
import logging
from pathlib import Path

import tensorflow as tf

from source.federated_devices import (ClientVirtualSequential,
                                      ClientVirtualModel,
                                      ServerSequential,
                                      ServerModel)
from source.fed_process import FedProcess
from source.utils import avg_dict
from source.constants import ROOT_LOGGER_STR

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
            model = self.model_fn(client_model_class, cards_train[indx])
            self.clients.append(model)
        self.server = self.model_fn(server_model_class, 1)

    def fit(self, federated_train_data, num_rounds, clients_per_round,
            epochs_per_round, federated_test_data=None,
            tensorboard_updates=1, logdir=Path(), callbacks=None,
            train_size=None, test_size=None, hierarchical=False):

        train_log_dir = logdir / 'train'
        self.train_summary_writer = \
            tf.summary.create_file_writer(str(train_log_dir))
        self.test_summary_writer = tf.summary.create_file_writer(str(logdir))

        self.build(train_size, hierarchical)

        history_test = [None] * len(self.clients)
        max_accuracy = -1.0
        max_acc_round = None
        for round_i in range(num_rounds):

            deltas = []
            history_train = []

            clients_sampled = random.sample(self.clients_indx,
                                            clients_per_round)
            for indx in clients_sampled:
                self.clients[indx].receive_and_save_weights(self.server)
                self.clients[indx].renew_center()
                if round_i > 0 and self.fed_avg_init:
                    self.clients[indx].initialize_kernel_posterior()

                history_single = self.clients[indx].fit(
                    federated_train_data[indx],
                    verbose=0,
                    validation_data=federated_test_data[indx],
                    epochs=epochs_per_round,
                    callbacks=callbacks)

                history_train.append({key: history_single.history[key]
                                      for key in history_single.history.keys()
                                      if 'val' not in key})
                history_test[indx] = \
                    {key.replace('val_', ''): history_single.history[key]
                     for key in history_single.history.keys()
                     if 'val' in key}

                self.clients[indx].apply_damping(self.damping_factor)
                delta = self.clients[indx].compute_delta()
                deltas.append(delta)

            aggregated_deltas = self.aggregate_deltas_multi_layer(deltas)
            self.server.apply_delta(aggregated_deltas)
            avg_train = avg_dict(history_train,
                                 [train_size[client]
                                  for client in clients_sampled])
            avg_test = avg_dict(history_test, test_size)

            if avg_test['sparse_categorical_accuracy'] > max_accuracy:
                max_accuracy = avg_test['sparse_categorical_accuracy']
                max_acc_round = round_i

            logger.debug(f"round: {round_i}, "
                         f"avg_train: {avg_train}, "
                         f"avg_test: {avg_test},"
                         f"max accuracy so far: {max_accuracy} reached at "
                         f"round {max_acc_round}")

            if round_i % tensorboard_updates == 0:
                with self.train_summary_writer.as_default():
                    for key in avg_train.keys():
                        tf.summary.scalar(key, avg_train[key], step=round_i)
                with self.test_summary_writer.as_default():
                    for key in avg_test.keys():
                        tf.summary.scalar(key, avg_test[key], step=round_i)
                    tf.summary.scalar('max_sparse_categorical_accuracy',
                                      max_accuracy, step=round_i)

        for i, client in enumerate(self.clients):
            client.save_weights(str(logdir / f'weights_{i}.h5'))
