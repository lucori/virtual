import tensorflow as tf
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers.recurrent import _caching_device
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.eager import context
from tensorflow.python.framework import ops


RECURRENT_DROPOUT_WARNING_MSG = (
    'RNN `implementation=2` is not supported when `recurrent_dropout` is set. '
    'Using `implementation=1`.')


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


class LSTMCellCentered(tf.keras.layers.LSTMCell):

    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super(LSTMCellCentered, self).__init__(units,
                                               activation=activation,
                                               recurrent_activation=recurrent_activation,
                                               use_bias=use_bias,
                                               kernel_initializer=kernel_initializer,
                                               recurrent_initializer=recurrent_initializer,
                                               bias_initializer=bias_initializer,
                                               unit_forget_bias=unit_forget_bias,
                                               kernel_regularizer=None,
                                               recurrent_regularizer=None,
                                               bias_regularizer=None,
                                               kernel_constraint=kernel_constraint,
                                               recurrent_constraint=recurrent_constraint,
                                               bias_constraint=bias_constraint,
                                               dropout=dropout,
                                               recurrent_dropout=recurrent_dropout,
                                               implementation=implementation,
                                               **kwargs)
        self.kernel_regularizer = kernel_regularizer()
        self.recurrent_regularizer = recurrent_regularizer()
        self.bias_regularizer = bias_regularizer()

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        input_dim = input_shape[-1]
        self.kernel_regularizer.center = self.add_weight('kernel_center',
                                                         shape=(input_dim, self.units * 4),
                                                         initializer=tf.keras.initializers.constant(0.),
                                                         dtype=self.dtype,
                                                         trainable=False)
        self.recurrent_regularizer.center = self.add_weight('recurrent_kernel_center',
                                                         shape=(self.units, self.units * 4),
                                                         initializer=tf.keras.initializers.constant(0.),
                                                         dtype=self.dtype,
                                                         trainable=False)
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device)

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                        ])
            else:
                bias_initializer = self.bias_initializer
            self.bias_regularizer.center = self.add_weight('bias_center',
                                                           shape=(self.units * 4,),
                                                           initializer=tf.keras.initializers.constant(0.),
                                                           dtype=self.dtype,
                                                           trainable=False)
            self.bias = self.add_weight(
                          shape=(self.units * 4,),
                          name='bias',
                          initializer=bias_initializer,
                          regularizer=self.bias_regularizer,
                          constraint=self.bias_constraint,
                          caching_device=default_caching_device)
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


class RNNCentered(tf.keras.layers.RNN):

    def receive_and_save_weights(self, layer):
        self.cell.receive_and_save_weights(layer.cell)

    def compute_delta(self):
        return self.cell.compute_delta()

    def apply_delta(self, delta):
        self.cell.apply_delta(delta)


class EmbeddingCentered(tf.keras.layers.Embedding):

    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        super(EmbeddingCentered, self).__init__(input_dim, output_dim,
                                                embeddings_initializer=embeddings_initializer,
                                                embeddings_regularizer=None,
                                                activity_regularizer=activity_regularizer,
                                                embeddings_constraint=embeddings_constraint,
                                                mask_zero=mask_zero,
                                                input_length=input_length,
                                                **kwargs)
        self.embeddings_regularizer = embeddings_regularizer()

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if context.executing_eagerly() and context.context().num_gpus():
            with ops.device('cpu:0'):
                self.embeddings_regularizer.center = self.add_weight(
                    shape=(self.input_dim, self.output_dim),
                    name='embeddings_center',
                    initializer=tf.keras.initializers.constant(0.),
                    dtype=self.dtype,
                    trainable=False),
                self.embeddings = self.add_weight(
                    shape=(self.input_dim, self.output_dim),
                    initializer=self.embeddings_initializer,
                    name='embeddings',
                    regularizer=self.embeddings_regularizer,
                    constraint=self.embeddings_constraint)
        else:
            self.embeddings_regularizer.center = self.add_weight(
                shape=(self.input_dim, self.output_dim),
                name='embeddings_center',
                initializer=tf.keras.initializers.constant(0.),
                dtype=self.dtype,
                trainable=False),
            self.embeddings = self.add_weight(
              shape=(self.input_dim, self.output_dim),
              initializer=self.embeddings_initializer,
              name='embeddings',
              regularizer=self.embeddings_regularizer,
              constraint=self.embeddings_constraint)
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
