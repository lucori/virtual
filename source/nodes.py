import tensorflow as tf
from dense_reparametrization_shared import DenseReparametrizationShared
from centered_l2_regularizer import DenseCentered


class Client(tf.keras.Sequential):

    def __init__(self, layers=None, name=None, num_samples=1):
        super(Client, self).__init__(layers=layers, name=name)
        self.s_i_to_update = False
        self.num_samples = num_samples

    def compute_delta(self):
        '''return a list of delta, one delta (loc and scale tensors) per layer'''

        delta = []
        for layer in self.layers:
            if isinstance(layer, DenseReparametrizationShared):
                delta.append(layer.compute_delta())
        return delta

    def renew_s_i(self):
        if self.s_i_to_update:
            for layer in self.layers:
                if isinstance(layer, DenseReparametrizationShared):
                    layer.renew_s_i()
        else:
            self.s_i_to_update = True

    def receive_s(self, server):
        for lc, ls in zip(self.layers, server.layers):
            if isinstance(lc, DenseReparametrizationShared) and isinstance(ls, DenseReparametrizationShared):
                lc.receive_s(ls)

    def apply_damping(self, damping_factor):
        for layer in self.layers:
            if isinstance(layer, DenseReparametrizationShared):
                layer.apply_damping(damping_factor)

    def initialize_kernel_posterior(self):
        for layer in self.layers:
            if isinstance(layer, DenseReparametrizationShared):
                layer.initialize_kernel_posterior()

    def call(self, inputs, training=None, mask=None):
        if self.num_samples > 1:
            sampling = MultiSampleEstimator(self, self.num_samples)
        else:
            sampling = super(Client, self).call
        output = sampling(inputs, training, mask)
        return output


class MultiSampleEstimator(tf.keras.layers.Layer):

    def __init__(self, model, num_samples):
        super(MultiSampleEstimator, self).__init__()
        self.model = model
        self.num_samples = num_samples

    def call(self, inputs, training=None, mask=None):
        output = []
        for _ in range(self.num_samples):
            output.append(super(Client, self.model).call(inputs, training, mask))
        output = tf.stack(output)
        output = tf.math.reduce_mean(output, axis=0)
        return output


class Server(tf.keras.Sequential):

    def __init__(self, layers=None, name=None):
        super(Server, self).__init__(layers=layers, name=name)

    def apply_delta(self, delta):
        for i, layer in enumerate(x for x in self.layers if isinstance(x, DenseReparametrizationShared)):
            if isinstance(layer, DenseReparametrizationShared):
                layer.apply_delta(delta[i])


class ClientFedProx(tf.keras.Sequential):

    def __init__(self, layers=None, name=None):
        super(ClientFedProx, self).__init__(layers=layers, name=name)
        self.old_weights = []

    def compute_delta(self):
        delta = []
        for layer in self.layers:
            if isinstance(layer, DenseCentered):
                delta.append(layer.compute_delta())
        return delta

    def receive_and_save_weights(self, server):
        for l_c, l_s in zip(self.layers, server.layers):
            if isinstance(l_c, DenseCentered):
                l_c.receive_and_save_weights(l_s)


class ServerFedProx(tf.keras.Sequential):

    def __init__(self, layers=None, name=None):
        super(ServerFedProx, self).__init__(layers=layers, name=name)

    def apply_delta(self, delta):
        for i, layer in enumerate(x for x in self.layers if isinstance(x, DenseCentered)):
            if isinstance(layer, DenseCentered):
                layer.apply_delta(delta[i])
