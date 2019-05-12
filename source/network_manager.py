import tensorflow as tf
from utils import get_posterior_from_layer, get_refined_prior, gaussian_ratio_par, Gate, LateralConnection
from general_utils import clone
from tensorflow.python.keras.engine.input_layer import InputLayer
from client import Client
from collections import defaultdict
import numpy as np


class NetworkManager:

    def __init__(self, server, data_set_size=None, n_samples=10):
        self.server = server
        self.clients = []
        self.optimizer = None
        self.compile_conf = {}
        self.data_set_size = data_set_size
        self.n_samples = n_samples
        self.num_clients = None

    def compile(self, optimizer, **kwargs):
        self.optimizer = dict(tf.keras.optimizers.serialize(optimizer))
        self.compile_conf = kwargs

    def fit(self, sequence, **kwargs):
        x = kwargs.pop('x')
        y = kwargs.pop('y', None)
        validation_data = kwargs.pop('validation_data', None)
        test_data = kwargs.pop('test_data', None)
        if validation_data:
            validation_data = list(validation_data)
        if test_data:
            test_data = list(test_data)
        unique, counts = np.unique(sequence, return_counts=True)
        counts = dict(zip(unique, counts))
        refined = [counts[u] > 1 for u in unique]
        history = defaultdict(list)
        evaluate = defaultdict(list)
        for t, i in enumerate(sequence):
            print('step ', t+1, ' in sequence of lenght ', len(sequence), ' task ', i+1)
            if t > 0:
                self.clients[i].update_prior(client_refining=refined[i])
                self.clients[i].set_q()
            optimizer = tf.keras.optimizers.deserialize(dict(self.optimizer))
            self.clients[i].compile(optimizer, **self.compile_conf)
            fit_config = {'x': x[i]}
            if y:
                fit_config['y'] = y[i]
            if validation_data:
                fit_config['validation_data'] = validation_data[i]
            fit_config.update(kwargs)
            hist = self.clients[i].fit(**fit_config)
            history[i].append(hist.history)
            if test_data:
                eval = self.clients[i].evaluate(*test_data[i])
                evaluate[i].append(eval)
                print(eval)
            if refined[i]:
                self.clients[i].new_t()

        return history, evaluate

    def create_clients(self, num_clients):
        self.num_clients = num_clients
        clients = []
        for _ in range(self.num_clients):
            client = self.client_from_server(self.server)
            clients.append(client)
        self.clients = clients
        return clients

    def client_from_server(self, server):
        input_tensor = server.input
        x = input_tensor
        server.client_count += 1
        name_suffix = '_client_' + str(server.client_count)
        for layer in server.layers:
            if not isinstance(layer, InputLayer):
                if 'lateral' in layer.name:
                    name = layer.name + name_suffix
                    x = LateralConnection(layer, self.data_set_size, self.n_samples, name=name)([x, layer.output])
                else:
                    x = clone(layer, data_set_size=self.data_set_size, n_samples=self.n_samples, name=name_suffix)(x)
        client = Client(input_tensor, x, n_samples=self.n_samples)
        client.data_set_size = self.data_set_size
        client.num_clients = self.num_clients
        return client

    def summary(self):
        for c in self.clients:
            c.summary()
