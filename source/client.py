import tensorflow as tf
from tensorflow_probability.python.layers import DenseReparameterization
from utils import get_posterior_from_layer, clone, sparse_array, get_refined_prior, gaussian_ratio_par
import tensorflow_probability as tfp


class _Client(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(_Client, self).__init__(*args, **kwargs)
        self.prior_fn_server = get_refined_prior
        self.prior_fn_client = None
        self.old_server_par = {}
        self.data_set_size = None
        self.n_samples = None

    def update_prior(self, client_refining=False, data_set=None):
        if client_refining:
            self.prior_fn_client = lambda l: tfp.layers.default_multivariate_normal_fn
            if data_set not in self.old_server_par:
                self.old_server_par[data_set] = None
        else:
            self.prior_fn_client = get_posterior_from_layer

        for layer in self.layers:
            if issubclass(layer.__class__, DenseReparameterization):
                if '_client_' not in layer.name:
                    old_weights = None
                    if client_refining:
                        if self.old_server_par[data_set]:
                            old_weights = self.old_server_par[data_set][layer.name]
                    layer.update_prior(self.prior_fn_server(layer, old_weights))
                else:
                    layer.update_prior(self.prior_fn_client(layer))


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

    def update_prior(self, client_refining=False, data_set=None):
        self.model.update_prior(client_refining, data_set)

    @property
    def data_set_size(self):
        return self.model.data_set_size

    @property
    def old_server_par(self):
        return self.model.old_server_par

    @data_set_size.setter
    def data_set_size(self, a):
        self.model.data_set_size = a