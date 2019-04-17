import tensorflow as tf
import tensorflow_probability as tfp


class Server(tf.keras.Sequential):
    def __init__(self, *args):
        super(Server, self).__init__(*args)
        self.prior_fn = lambda layer: tfp.layers.default_multivariate_normal_fn
        self.client_count = -1

    def get_dict_weights(self):
        return {layer.name: layer.get_weights() for layer in self.layers}
