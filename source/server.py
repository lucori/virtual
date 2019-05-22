import tensorflow as tf
from utils import get_refined_prior
from tensorflow_probability.python.layers import DenseReparameterization, DenseFlipout


class Server(tf.keras.Sequential):

    def __init__(self, *args, data_set_size, n_samples):
        super(Server, self).__init__(*args)
        self.data_set_size = data_set_size
        self.n_samples = n_samples

    def update_prior(self, q, t):
        for layer in self.layers:
            if issubclass(layer.__class__, DenseReparameterization) or \
                    issubclass(layer.__class__, DenseFlipout):
                layer.update_divergence(self.data_set_size*self.n_samples)
                if q:
                    layer.update_prior(get_refined_prior(q[layer.name], t[layer.name]))
                layer.reparametrize_posterior()

    def get_t(self):
        return {layer.name: layer.get_t() for layer in self.layers
                if issubclass(layer.__class__, DenseReparameterization) or
                issubclass(layer.__class__, DenseFlipout)}

    def get_q(self):
        return {layer.name: layer.get_weights() for layer in self.layers
                if issubclass(layer.__class__, DenseReparameterization) or
                issubclass(layer.__class__, DenseFlipout)}
