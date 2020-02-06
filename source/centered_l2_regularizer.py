import tensorflow as tf
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
import numpy as np
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


def aggregate_deltas_multi_layer_avg(deltas, train_size_ratio):
    '''deltas is a list of delta, every element is a list of delta per layer'''
    aggregated_deltas = []
    deltas = list(map(list, zip(*deltas)))
    for delta_layer in deltas:
        aggregated_deltas.append(aggregate_deltas_single_layer_avg(delta_layer, train_size_ratio))
    return aggregated_deltas


def aggregate_deltas_single_layer_avg(deltas, train_size_ratio):
    deltas = list(zip(*deltas))
    aggregated = []
    for delta in deltas:
        aggregated.append(tf.math.add_n([d*ratio for d, ratio in zip(delta, train_size_ratio)]))
    return aggregated


@tf.keras.utils.register_keras_serializable(package='Custom')
class CenteredL2Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l2=0.):
        self.l2 = l2
        self.center = None

    def __call__(self, x):
        return self.l2 * tf.math.reduce_sum(tf.math.square(x - self.center))

    def get_config(self):
        return {'l2': float(self.l2), 'center': float(self.center.numpy())}


class DenseCentered(tf.keras.layers.Dense):

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(DenseCentered, self).__init__(units,
                 activation=activation,
                 use_bias=use_bias,
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint,
                 bias_constraint=bias_constraint,
                 **kwargs)

        self.kernel_regularizer = kernel_regularizer()
        self.bias_regularizer = bias_regularizer()

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: last_dim})
        self.kernel_regularizer.center = self.add_weight('kernel_center',
                                                         shape=[last_dim, self.units],
                                                         initializer=tf.keras.initializers.constant(0.),
                                                         dtype=self.dtype,
                                                         trainable=False)
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias_regularizer.center = self.add_weight('bias_center',
                                                 shape=[self.units,],
                                                 initializer=tf.keras.initializers.constant(0.),
                                                 dtype=self.dtype,
                                                 trainable=False)
            self.bias = self.add_weight(
              'bias',
              shape=[self.units,],
              initializer=self.bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint,
              dtype=self.dtype,
              trainable=True)
        else:
            self.bias = None
        self.built = True

    def receive_and_save_weights(self, layer):
        for v_c, c_c, v_s in zip(self.trainable_variables, self.non_trainable_variables, layer.trainable_variables):
            v_c.assign(v_s.numpy())
            c_c.assign(v_s.numpy())

    def compute_delta(self):
        delta = [v - c for v, c in zip(self.trainable_variables, self.non_trainable_variables)]
        return tuple(delta)

    def apply_delta(self, delta):
        for v, d in zip(self.trainable_variables, delta):
            v.assign_add(d)