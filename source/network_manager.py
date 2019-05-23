import tensorflow as tf
from layers import Gate, LateralConnection
from general_utils import clone, timeit_context, new_session
from tensorflow.python.keras.engine.input_layer import InputLayer
from client import Client
from collections import defaultdict
import numpy as np
from utils import softminus


class NetworkManager:

    def __init__(self, server, data_set_size=None, n_samples=10, num_clients=None, sess_config=None, run_obj=None,
                 method=None):
        self.server = server
        self.clients = []
        self.optimizer = None
        self.compile_conf = {}
        self.data_set_size = data_set_size
        self.n_samples = n_samples
        if num_clients:
            self.num_clients = num_clients
        elif data_set_size:
            self.num_clients = len(data_set_size)
        else:
            self.num_clients = None
        self.create_clients()
        self.server.data_set_size= data_set_size
        self.server_variational_layer_names = [layer.name for layer in self.server.layers
                                               if '_client_' not in layer.name and len(layer.weights) > 1]
        self.initialize_t()
        self.q = None
        self.sess_config = sess_config
        self.run = run_obj
        self.method = method

    def compile(self, optimizer, **kwargs):
        self.optimizer = dict(tf.keras.optimizers.serialize(optimizer))
        self.compile_conf = kwargs
        optimizer = tf.keras.optimizers.deserialize(dict(self.optimizer))
        [cl.compile(optimizer, **self.compile_conf) for cl in self.clients]

    def fit(self, sequence=None, **kwargs):
        x = kwargs.pop('x')
        y = kwargs.pop('y', None)
        validation_data = kwargs.pop('validation_data', None)
        test_data = kwargs.pop('test_data', None)
        steps_per_epoch = kwargs.pop('steps_per_epoch', None)
        validation_steps = kwargs.pop('validation_steps', None)
        if validation_data:
            validation_data = list(validation_data)
        if test_data:
            test_data = list(test_data)
        history = defaultdict(list)
        evaluate = defaultdict(list)
        fit_config = []
        for i in sequence:
            fit_config.append({})
            fit_config[i] = {'x': x[i]}
            if y:
                fit_config[i]['y'] = y[i]
            if validation_data:
                fit_config[i]['validation_data'] = validation_data[i]
            #if steps_per_epoch:
            #fit_config[i]['steps_per_epoch'] = None
            #if validation_steps:
            #fit_config[i]['validation_steps'] = None
            fit_config[i].update(kwargs)

        for t, i in enumerate(sequence):
            print('step ', t+1, ' in sequence of lenght ', len(sequence), ' task ', i+1)
            print('updating prior')
            with timeit_context('prior update'):
                self.server.update_prior(i, self.q, self.t[i])
            print('prior updated')
            #optimizer = tf.keras.optimizers.deserialize(dict(self.optimizer))
            print('compiling client')
            #self.clients[i].compile(optimizer, **self.compile_conf)
            print('client compiled')

            with timeit_context('fit'):
                hist = self.clients[i].fit(x[i], y[i], epochs=2)
            history[i].append(hist.history)
            if test_data:
                eval = self.clients[i].evaluate(*test_data[i])
                evaluate[i].append(eval)
                print(eval)
                if self.run:
                    self.run.log_scalar('accuracy_' + self.method + '_task_' + str(i+1), eval[-1])
            print('computing new t')
            with timeit_context('t and q'):
                self.t[i] = self.server.get_t()
                self.q = self.server.get_q()
            print('new t computed')

        return history, evaluate

    def client_from_server(self, indx):
        input_tensor = self.server.input
        x = input_tensor
        name_suffix = '_client_'
        for layer in self.server.layers:
            if not isinstance(layer, InputLayer):
                if 'lateral' in layer.name:
                    name = layer.name + name_suffix
                    x = LateralConnection(layer, self.data_set_size[indx], self.n_samples, name=name)([x, layer.output])
                else:
                    x = clone(layer, data_set_size=self.data_set_size[indx], n_samples=self.n_samples,
                              name=name_suffix)(x)
        client = Client(input_tensor, x, n_samples=self.n_samples)
        return client

    def create_clients(self):
        for i, _ in enumerate(self.data_set_size):
            self.clients.append(self.client_from_server(i))

    def initialize_t(self):
        def standard_normal(layer):
            shape = layer.weights[0].shape.as_list()
            return [np.zeros(shape, dtype=np.float32),
                    softminus(np.sqrt(self.num_clients - 1) * np.ones(shape, dtype=np.float32))]

        self.t = [{layer_name: standard_normal(self.server.get_layer(layer_name))
                   for layer_name in self.server_variational_layer_names} for _ in range(self.num_clients)]

