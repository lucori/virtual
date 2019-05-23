import tensorflow as tf
from utils import gaussian_ratio_par
from tensorflow_probability.python.layers import DenseReparameterization


class Server(tf.keras.Sequential):

    def __init__(self, *args, data_set_size, n_samples):
        super(Server, self).__init__(*args)
        self.data_set_size = data_set_size
        self.n_samples = n_samples

    def update_prior(self, i, q, t):
        for layer in self.layers:
            if issubclass(layer.__class__, DenseReparameterization):
                layer.set_data_set_size(self.data_set_size[i]*self.n_samples)
                if q:
                    layer.update_prior(gaussian_ratio_par(q[layer.name], t[layer.name]))

    def get_t(self):
        return {layer.name: layer.get_t() for layer in self.layers
                if issubclass(layer.__class__, DenseReparameterization)}

    def get_q(self):
        return {layer.name: layer.get_weights() for layer in self.layers
                if issubclass(layer.__class__, DenseReparameterization)}
