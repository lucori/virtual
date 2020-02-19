import random
from federated_devices import ClientSequential, ServerSequential
from centered_layers import aggregate_deltas_multi_layer_avg
import tensorflow as tf
from utils import avg_dict, avg_dict_eval


class FedProx:

    def __init__(self, model_fn, num_clients):

        self.model_fn = model_fn
        self.num_clients = num_clients
        self.clients_indx = range(self.num_clients)
        self.server = None
        self.client = None
        self.build()

    def build(self):
        self.client = self.model_fn(ClientSequential, 1)
        self.server = self.model_fn(ServerSequential, 1)

    def fit(self, federated_train_data, num_rounds, clients_per_round, epochs_per_round, federated_test_data=None,
            tensorboard_updates=1, logdir='', callbacks=None, train_size=None, test_size=None):

        train_log_dir = logdir + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(logdir)

        for round in range(num_rounds):

            deltas = []
            history_train = []

            clients_sampled = random.sample(self.clients_indx, clients_per_round)
            for indx in clients_sampled:
                self.client.receive_and_save_weights(self.server)
                history_single = self.client.fit(federated_train_data[indx], verbose=0,
                                                 epochs=epochs_per_round, callbacks=callbacks)
                history_train.append({key: history_single.history[key] for key in history_single.history.keys()
                                      if 'val' not in key})
                delta = self.client.compute_delta()
                deltas.append(delta)

            aggregated_deltas = aggregate_deltas_multi_layer_avg(deltas,
                                                                 [train_size[client]/sum(train_size)
                                                                  for client in clients_sampled])
            self.server.apply_delta(aggregated_deltas)
            test = [self.server.evaluate(test_data, verbose=0) for test_data in federated_test_data]
            avg_train = avg_dict(history_train, [train_size[client] for client in clients_sampled])
            avg_test = avg_dict_eval(test, [size/sum(test_size) for size in test_size])
            print('round:', round, avg_train, avg_test)
            if round % tensorboard_updates == 0:
                for i, key in enumerate(avg_train.keys()):
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar(key, avg_train[key], step=round)
                    with self.test_summary_writer.as_default():
                        tf.summary.scalar(key, avg_test[i], step=round)
