import tensorflow as tf
from dense_reparametrization_shared import DenseReparametrizationShared


class Client(tf.keras.Sequential):

    def __init__(self, layers=None, name=None, num_samples=1):
        super(Client, self).__init__(layers=layers, name=name)
        self.s_i_to_update = False
        self.num_samples = num_samples

    def compute_delta(self):
        delta = []
        for layer in self.layers:
            if hasattr(layer, 'compute_delta'):
                delta.append(layer.compute_delta())
        return delta

    def receive_and_save_weights(self, server):
        for l_c, l_s in zip(self.layers, server.layers):
            if hasattr(l_c, 'receive_and_save_weights'):
                l_c.receive_and_save_weights(l_s)


class Server(tf.keras.Sequential):

    def __init__(self, layers=None, name=None):
        super(Server, self).__init__(layers=layers, name=name)

    def apply_delta(self, delta):
        for i, layer in enumerate(x for x in self.layers if hasattr(x, 'apply_delta')):
            if hasattr(layer, 'apply_delta'):
                layer.apply_delta(delta[i])


class ClientVirtual(Client):

    def renew_s_i(self):
        if self.s_i_to_update:
            for layer in self.layers:
                if isinstance(layer, DenseReparametrizationShared):
                    layer.renew_s_i()
        else:
            self.s_i_to_update = True

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
