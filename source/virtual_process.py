import random
from nodes import Client, Server
from tfp_utils import aggregate_deltas_multi_layer
import tensorflow as tf
import datetime


def avg_dict(history_list, cards):
    avg_dict = {}
    keys = history_list[0].keys()
    for key in keys:
        avg_dict[key] = sum([history[key][-1]*card for history, card in zip(history_list, cards)])/sum(cards)
    return avg_dict


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/virtual_' + current_time + '/train'
test_log_dir = 'logs/virtual_' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


class VirtualFedProcess:

    def __init__(self, model_fn, num_clients, damping_factor=1):

        self.model_fn = model_fn
        self.num_clients = num_clients
        self.damping_factor = damping_factor
        self.clients_indx = range(self.num_clients)
        self.clients = []
        self.server = None

    def build(self, cards_train):
        for indx in self.clients_indx:
            model = self.model_fn(Client, cards_train[indx])
            self.clients.append(model)
        self.server = self.model_fn(Server)

    def fit(self, federated_train_data, num_rounds, clients_per_round, federated_test_data=None):

        cards_train = [100*tf.data.experimental.cardinality(data).numpy() for data in federated_train_data]
        cards_test = [tf.data.experimental.cardinality(data).numpy() for data in federated_test_data]

        self.build(cards_train)
        for round in range(num_rounds):

            deltas = []
            history_train = []
            history_test = []

            clients_sampled = random.sample(self.clients_indx, clients_per_round)
            for indx in clients_sampled:
                self.clients[indx].receive_s(self.server)
                self.clients[indx].renew_s_i()
                if round > 0:
                    self.clients[indx].initialize_kernel_posterior()
                history_single = self.clients[indx].fit(federated_train_data[indx], verbose=0,
                                                        validation_data=federated_test_data[indx])
                history_train.append({key: history_single.history[key] for key in history_single.history.keys()
                                      if 'val' not in key})
                history_test.append({key.replace('val_', ''): history_single.history[key] for key in
                                     history_single.history.keys() if 'val' in key})
                self.clients[indx].apply_damping(self.damping_factor)
                delta = self.clients[indx].compute_delta()
                deltas.append(delta)

            aggregated_deltas = aggregate_deltas_multi_layer(deltas)
            self.server.apply_delta(aggregated_deltas)
            avg_train = avg_dict(history_train, [cards_train[client] for client in clients_sampled])
            avg_test = avg_dict(history_test, [cards_test[client] for client in clients_sampled])
            print('round:', round, avg_train, avg_test)
            with train_summary_writer.as_default():
                for key in avg_train.keys():
                    tf.summary.scalar(key, avg_train[key], step=round)
            with test_summary_writer.as_default():
                for key in avg_test.keys():
                    tf.summary.scalar(key, avg_test[key], step=round)
