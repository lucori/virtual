import tensorflow as tf


class _Client:

    def compute_delta(self):
        delta = []
        for layer in self.layers:
            if hasattr(layer, 'compute_delta'):
                delta.append(layer.compute_delta())
        return delta

    def renew_center(self, center_to_update=True):
        for layer in self.layers:
            if hasattr(layer, 'renew_center'):
                layer.renew_center(center_to_update)

    def receive_and_save_weights(self, server):
        for l_c, l_s in zip(self.layers, server.layers):
            if hasattr(l_c, 'receive_and_save_weights'):
                l_c.receive_and_save_weights(l_s)


class _Server:

    def apply_delta(self, delta):
        for i, layer in enumerate(x for x in self.layers if hasattr(x, 'apply_delta')):
            if hasattr(layer, 'apply_delta'):
                layer.apply_delta(delta[i])


class _ClientVirtual(_Client):

    def apply_damping(self, damping_factor):
        for layer in self.layers:
            if hasattr(layer, 'apply_damping'):
                layer.apply_damping(damping_factor)

    def initialize_kernel_posterior(self):
        for layer in self.layers:
            if hasattr(layer, 'initialize_kernel_posterior'):
                layer.initialize_kernel_posterior()

    def call(self, inputs, training=None, mask=None):
        if self.num_samples > 1:
            sampling = MultiSampleEstimator(self, self.num_samples)
        else:
            sampling = super(_ClientVirtual, self).call
        output = sampling(inputs, training, mask)
        return output


class ClientSequential(tf.keras.Sequential, _Client):

    def __init__(self, layers=None, name=None, num_samples=1):
        super(ClientSequential, self).__init__(layers=layers, name=name)
        self.num_samples = num_samples


class ClientModel(tf.keras.Model, _Client):

    def __init__(self, *args, **kwargs):
        self.num_samples = kwargs.pop('num_samples', 1)
        super(ClientModel, self).__init__(*args, **kwargs)


class ServerSequential(tf.keras.Sequential, _Server):

    def __init__(self, layers=None, name=None, num_samples=1):
        super(ServerSequential, self).__init__(layers=layers, name=name)
        self.num_samples = num_samples


class ServerModel(tf.keras.Model, _Server):

    def __init__(self, *args, **kwargs):
        self.num_samples = kwargs.pop('num_samples', 1)
        super(ServerModel, self).__init__(*args, **kwargs)


class ClientVirtualSequential(ClientSequential, _ClientVirtual):
    pass


class ClientVirtualModel(ClientModel, _ClientVirtual):
    pass


class MultiSampleEstimator(tf.keras.layers.Layer):

    def __init__(self, model, num_samples):
        super(MultiSampleEstimator, self).__init__()
        self.model = model
        self.num_samples = num_samples

    def call(self, inputs, training=None, mask=None):
        output = []
        for _ in range(self.num_samples):
            output.append(super(_ClientVirtual, self.model).call(inputs, training, mask))
        output = tf.stack(output)
        output = tf.math.reduce_mean(output, axis=0)
        return output
