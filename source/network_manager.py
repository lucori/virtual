import tensorflow as tf
from utils import get_posterior_from_layer, clone, get_refined_prior, gaussian_ratio_par, Gate
from tensorflow.python.keras.engine.input_layer import InputLayer
from client import Client
from collections import defaultdict


class NetworkManager:

    def __init__(self, server, data_set_size=None, n_samples=10):
        self.server = server
        self.clients = []
        self.optimizer = None
        self.compile_conf = {}
        self.data_set_size = data_set_size
        self.n_samples = n_samples

    def compile(self, optimizer, **kwargs):
        self.optimizer = dict(tf.keras.optimizers.serialize(optimizer))
        self.compile_conf = kwargs

    def fit(self, model_sequence, data_sequence, **kwargs):
        x = kwargs.pop('x')
        y = kwargs.pop('y', None)
        validation_data = kwargs.pop('validation_data', None)
        test_data = kwargs.pop('test_data', None)
        if validation_data:
            validation_data = list(validation_data)
        if test_data:
            test_data = list(test_data)
        optimizer = tf.keras.optimizers.deserialize(dict(self.optimizer))
        sequence = list(zip(model_sequence, data_sequence))
        refined = list(set([x for x in sequence if sequence.count(x) >= 2]))
        history = defaultdict(list)
        evaluate = defaultdict(list)
        for t, (i, j) in enumerate(sequence):
            print('step ', t, ' in sequence of lenght ', len(sequence), ' task ', i)
            self.clients[i].update_prior(client_refining=((i, j) in refined), data_set=j)
            self.clients[i].compile(optimizer, **self.compile_conf)
            fit_config = {'x': x[j]}
            if y:
                fit_config['y'] = y[j]
            if validation_data:
                fit_config['validation_data'] = validation_data[j]
            fit_config.update(kwargs)
            hist = self.clients[i].fit(**fit_config)
            history[i].append(hist.history)
            if test_data:
                eval = self.clients[i].evaluate(*test_data[j])
                evaluate[i].append(eval)
                print(eval)
            if (i, j) in refined:
                if self.clients[i].old_server_par[j]:
                    for layer in self.clients[i].old_server_par[j]:
                        self.clients[i].old_server_par[j][layer] = gaussian_ratio_par(
                                                self.server.get_layer(layer).get_weights(),
                                                self.clients[i].old_server_par[j][layer])
                else:
                    self.clients[i].old_server_par[j] = self.server.get_dict_weights()
        return history, evaluate

    def create_clients(self, num_clients):
        clients = []
        for _ in range(num_clients):
            client = self.client_from_server(self.server)
            clients.append(client)
        self.clients = clients
        return clients

    def client_from_server(self, server):
        input_tensor = server.input
        x = input_tensor
        server.client_count += 1
        name = '_client_' + str(server.client_count)
        for layer in server.layers:
            if not isinstance(layer, InputLayer):
                if 'lateral' in layer.name:
                    out1 = clone(layer, data_set_size=self.data_set_size, n_samples=self.n_samples,
                                 activation='linear', name=name + '_from_client')(x)
                    out2 = clone(layer, data_set_size=self.data_set_size, n_samples=self.n_samples,
                                 name=name + '_from_server1')(layer.output)
                    out2 = clone(layer, data_set_size=self.data_set_size, n_samples=self.n_samples,
                                 activation='linear', name=name + '_from_server2')(out2)
                    out2 = Gate()(out2)
                    x = tf.keras.layers.add([out1, out2], name=layer.name + '_add' + name)
                    x = tf.keras.layers.Activation(layer.get_config()['activation'],
                                                   name=layer.name + '_activation' + name)(x)
                else:
                    x = clone(layer, data_set_size=self.data_set_size, n_samples=self.n_samples, name=name)(x)
        client = Client(input_tensor, x, n_samples=self.n_samples)
        client.data_set_size = self.data_set_size
        return client

    def summary(self):
        for c in self.clients:
            c.summary()
