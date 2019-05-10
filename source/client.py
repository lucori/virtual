import tensorflow as tf
from tensorflow_probability.python.layers import DenseReparameterization
from utils import get_posterior_from_layer, get_refined_prior, gaussian_ratio_par, LateralConnection, prior_wrapper, gaussian_prod_par
import tensorflow_probability as tfp
from utils import softminus
import numpy as np


class _Client(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(_Client, self).__init__(*args, **kwargs)
        self.prior_fn_server = get_refined_prior
        self.prior_fn_client = None
        self.data_set_size = None
        self.n_samples = None
        self.num_clients = None
        self.server_variational_layers = [layer for layer in self.layers
                                          if '_client_' not in layer.name and len(layer.weights) > 1]
        for layer in self.server_variational_layers:
            layer.reparametrize_posterior()

    def update_prior(self, client_refining=False):
        if client_refining:
            self.prior_fn_client = lambda l: tfp.layers.default_multivariate_normal_fn
        else:
            self.prior_fn_client = get_posterior_from_layer

        for layer in self.layers:
            if issubclass(layer.__class__, DenseReparameterization) or isinstance(layer, LateralConnection):
                if layer in self.server_variational_layers:
                    layer.update_prior(self.prior_fn_server(layer, self.t[layer.name]))
                else
                    layer.update_prior(prior_wrapper(self.prior_fn_client, layer))

    def new_t_old(self):
        self.t = {layer.name: gaussian_ratio_par(gaussian_prod_par(layer.get_weights(), self.t[layer.name]),
                                                 self.q[layer.name]) for layer in self.server_variational_layers}

    def new_t(self):
        self.t = {layer.name: layer.get_t() for layer in self.server_variational_layers}

    def set_q(self):
        self.q = {layer.name: layer.get_weights() for layer in self.server_variational_layers}

    def initialize_q(self):
        def standard_normal(layer):
            shape = layer.weights[0].shape.as_list()
            return [np.zeros(shape, dtype=np.float32),
                    softminus(np.sqrt((self.num_clients - 1)/self.num_clients)*np.ones(shape, dtype=np.float32))]
        self.q = {layer.name: standard_normal(layer) for layer in self.server_variational_layers}

    def initialize_t(self):
        def standard_normal(layer):
            shape = layer.weights[0].shape.as_list()
            return [np.zeros(shape, dtype=np.float32),
                    softminus(np.sqrt(self.num_clients - 1)*np.ones(shape, dtype=np.float32))]
        self.t = {layer.name: standard_normal(layer) for layer in self.server_variational_layers}


class Client(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(Client, self).__init__()
        n_samples = kwargs.pop('n_samples', None)
        model = kwargs.pop('model', None)
        if n_samples:
            self.n_samples = n_samples
        else:
            self.n_samples = 10
        if model:
            self.model = model
        else:
            self.model = _Client(*args, **kwargs)
        self.model.n_samples = self.n_samples

    def call(self, inputs):
        output = []
        for _ in range(self.n_samples):
            output.append(self.model.call(inputs))

        output = tf.keras.layers.Lambda(lambda q: tf.stack(q))(output)
        output = tf.keras.layers.Lambda(lambda q: tf.reduce_sum(q, axis=0))(output)
        return output

    def update_prior(self, client_refining=False):
        self.model.update_prior(client_refining)

    def new_t(self):
        self.model.new_t()

    def set_q(self):
        self.model.set_q()

    @property
    def data_set_size(self):
        return self.model.data_set_size

    @property
    def num_clients(self):
        return self.model.num_clients

    @num_clients.setter
    def num_clients(self, num_clients):
        self.model.num_clients = num_clients
        self.model.initialize_q()
        self.model.initialize_t()

    @property
    def layers(self):
        return self.model.layers

    @property
    def t(self):
        return self.model.t

    @property
    def q(self):
        return self.model.q

    @t.setter
    def t(self, t):
        self.model.t = t

    @q.setter
    def q(self, q):
        self.model.q = q

    @data_set_size.setter
    def data_set_size(self, a):
        self.model.data_set_size = a
