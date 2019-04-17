import tensorflow as tf
from utils import get_posterior_from_layer, clone, sparse_array, get_refined_prior, gaussian_ratio_par
from tensorflow.python.keras.engine.input_layer import InputLayer
from client import Client


class NetworkManager:

    def __init__(self, server):
        self.server = server
        self.clients = []
        self.optimizer = None
        self.compile_conf = {}

    def compile(self, optimizer, **kwargs):
        self.optimizer = dict(tf.keras.optimizers.serialize(optimizer))
        self.compile_conf = kwargs

    def fit(self, model_sequence, data_sequence, **kwargs):
        x = kwargs.pop('x')
        y = kwargs.pop('y', None)
        validation_data = kwargs.pop('validation_data', None)
        if validation_data:
            validation_data = list(validation_data)
        optimizer = tf.keras.optimizers.deserialize(dict(self.optimizer))
        sequence = list(zip(model_sequence, data_sequence))
        refined = list(set([x for x in sequence if sequence.count(x) >= 2]))
        for (i, j) in sequence:
            self.server, self.clients[i] = self.clients[i].new_server_and_client(self.server,
                                                                                 client_refining=((i, j) in refined),
                                                                                 data_set=j)
            self.clients[i].compile(optimizer, **self.compile_conf)
            fit_config = {'x': x[j]}
            if y:
                fit_config['y'] = y[j]
            if validation_data:
                fit_config['validation_data'] = validation_data[j]
            fit_config.update(kwargs)
            self.clients[i].fit(**fit_config)
            if (i, j) in refined:
                if self.clients[i].old_server_par[j]:
                    for layer in self.clients[i].old_server_par[j]:
                        self.clients[i].old_server_par[j][layer] = gaussian_ratio_par(
                                                self.server.get_layer(layer).get_weights(),
                                                self.clients[i].old_server_par[j][layer])
                else:
                    self.clients[i].old_server_par[j] = self.server.get_dict_weights()

    def create_clients(self, num_clients):
        outputs = []
        clients = []
        for _ in range(num_clients):
            client = self.client_from_server(self.server)
            outputs.append(client.output)
            clients.append(client)
        self.clients = clients
        return clients

    def client_from_server(self, server):
        input_tensor = server.input
        x = input_tensor
        server.client_count += 1
        name = '_client_' + str(server.client_count)
        for layer in server._layers:
            if not isinstance(layer, InputLayer):
                if 'lateral' in layer.name:
                    out1 = clone(layer, activation='linear', name=name + '_from_client')(x)
                    out2 = clone(layer, activation='linear', name=name + '_from_server')(layer.output)
                    x = tf.keras.layers.add([out1, out2], name=layer.name + '_add' + name)
                    x = tf.keras.layers.Activation(layer.get_config()['activation'],
                                                   name=layer.name + '_activation' + name)(x)
                else:
                    x = clone(layer, name=name)(x)

        return Client(input_tensor, x)