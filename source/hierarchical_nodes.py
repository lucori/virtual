import tensorflow as tf
from dense_reparametrization_shared import DenseReparametrizationShared
from tensorflow.python.keras import initializers
from tensorflow.python.framework import tensor_shape


class Gate(tf.keras.layers.Layer):

    def __init__(self,
                 initializer=tf.keras.initializers.RandomUniform(minval=0,
                                                                 maxval=0.1),
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(Gate, self).__init__(**kwargs)
        self.initializer = initializers.get(initializer)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        self.gate = self.add_weight(
            'gate',
            shape=input_shape[1:],
            initializer=self.initializer,
            dtype=self.dtype,
            trainable=True)
        self.built = True

    def call(self, inputs):
        outputs = tf.math.multiply(inputs, self.gate)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'initializer': initializers.serialize(self.initializer),
            }
        base_config = super(Gate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ClientModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        self.num_samples = kwargs.pop('num_samples', 1)
        super(ClientModel, self).__init__(*args, **kwargs)
        self.s_i_to_update = False

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
        sampling = super(ClientModel, self).call
        output = sampling(inputs, training, mask)
        return output


class ServerModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        self.num_samples = kwargs.pop('num_samples', 1)
        super(ServerModel, self).__init__(*args, **kwargs)

    def apply_delta(self, delta):
        for i, layer in enumerate(x for x in self.layers if isinstance(x, DenseReparametrizationShared)):
            if isinstance(layer, DenseReparametrizationShared):
                layer.apply_delta(delta[i])


