import tensorflow as tf
from tensorflow_probability.python.layers import DenseReparameterization
from utils import get_posterior_from_layer, get_refined_prior, gaussian_ratio_par, LateralConnection, prior_wrapper
import tensorflow_probability as tfp
from utils import softminus
import numpy as np


class _Client(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(_Client, self).__init__(*args, **kwargs)
        self.prior_fn_server = get_refined_prior
        self.prior_fn_client = None
        self.q = None
        self.t = None
        self.data_set_size = None
        self.n_samples = None

    def update_prior(self, client_refining=False):
        if client_refining:
            self.prior_fn_client = lambda l: tfp.layers.default_multivariate_normal_fn
        else:
            self.prior_fn_client = get_posterior_from_layer

        for layer in self.layers:
            if issubclass(layer.__class__, DenseReparameterization) or isinstance(layer, LateralConnection):
                if '_client_' not in layer.name:
                    t = None
                    if client_refining:
                        if self.t:
                            t = self.t[layer.name]
                    layer.update_prior(self.prior_fn_server(layer, t))
                else:
                    layer.update_prior(prior_wrapper(self.prior_fn_client, layer))

    def new_t(self, server):
        if self.q:
            self.t = {layer.name: gaussian_ratio_par(layer.get_weights(), self.q[layer.name])
                      for layer in server.layers}
        else:
            self.t = {}
            for layer_name, layer_weight in server.get_dict_weights().items():
                if len(layer_weight) > 1:
                    standard_normal = [np.zeros_like(layer_weight[0]),
                                       softminus(np.ones_like(layer_weight[1]))]
                    self.t[layer_name] = gaussian_ratio_par(
                        layer_weight,
                        standard_normal)
                else:
                    self.t[layer_name] = []

    def set_q(self):
        self.q = {layer.name: layer.get_weights() for layer in self.layers if '_client_' not in layer.name}


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

    def new_t(self, server):
        self.model.new_t(server)

    def set_q(self):
        self.model.set_q()

    @property
    def data_set_size(self):
        return self.model.data_set_size

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