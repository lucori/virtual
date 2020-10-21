import tensorflow as tf
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers.recurrent import _caching_device
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.utils import conv_utils


RECURRENT_DROPOUT_WARNING_MSG = (
    'RNN `implementation=2` is not supported when `recurrent_dropout` is set. '
    'Using `implementation=1`.')


@tf.keras.utils.register_keras_serializable(package='Custom')
class CenteredL2Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l2=0.):
        self.l2 = l2
        self.center = None

    def __call__(self, x):
        return self.l2 * tf.math.reduce_sum(tf.math.square(x - self.center))

    def get_config(self):
        return {'l2': float(self.l2), 'center': float(self.center.numpy())}


class LayerCentered:

    def compute_delta(self):
        delta_dict = {}
        for key in self.client_variable_dict.keys():
            delta_dict[key] = (
                self.delta_function(self.client_variable_dict[key],
                                    self.client_center_variable_dict[key]))
        return delta_dict

    def renew_center(self, center_to_updated=True):
        if 'natural' in self.name or center_to_updated:
            for key in self.client_center_variable_dict.keys():
                self.client_center_variable_dict[key].assign(
                    self.client_variable_dict[key])

    def apply_delta(self, delta):
        for key in self.server_variable_dict.keys():
            add = self.apply_delta_function(
                self.server_variable_dict[key], delta[key])
            self.server_variable_dict[key].assign(add)
            self.client_variable_dict[key].assign(add)

    def receive_and_save_weights(self, layer_server):
        for key in self.server_variable_dict.keys():
            self.server_variable_dict[key].assign(
                layer_server.server_variable_dict[key])


class DenseCentered(tf.keras.layers.Dense, LayerCentered):

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

        super(DenseCentered, self).__init__(
            units,
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
        self.delta_function = tf.subtract
        self.apply_delta_function = tf.add
        self.client_variable_dict = {}
        self.server_variable_dict = {}
        self.client_center_variable_dict = {}

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                'Unable to build `Dense` layer with non-floating point '
                'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: last_dim})
        self.kernel_regularizer.center = self.add_weight(
            'kernel_center',
            shape=[last_dim, self.units],
            initializer=tf.keras.initializers.constant(0.),
            dtype=self.dtype,
            trainable=False)
        self.kernel = self.add_weight('kernel',
                                      shape=[last_dim, self.units],
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype=self.dtype,
                                      trainable=True)
        if self.use_bias:
            self.bias_regularizer.center = self.add_weight(
                'bias_center',
                shape=[self.units, ],
                initializer=tf.keras.initializers.constant(0.),
                dtype=self.dtype,
                trainable=False)

            self.bias = self.add_weight('bias',
                                        shape=[self.units, ],
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        else:
            self.bias = None

        self.client_variable_dict['kernel'] = self.kernel
        self.server_variable_dict['kernel'] = self.kernel
        self.client_center_variable_dict['kernel'] = \
            self.kernel_regularizer.center

        if self.use_bias:
            self.client_variable_dict['bias'] = self.bias
            self.server_variable_dict['bias'] = self.bias
            self.client_center_variable_dict['bias'] = \
                self.bias_regularizer.center

        self.built = True


class LSTMCellCentered(tf.keras.layers.LSTMCell, LayerCentered):

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
        super(LSTMCellCentered, self).__init__(
            units,
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
        self.delta_function = tf.subtract
        self.apply_delta_function = tf.add
        self.client_variable_dict = {}
        self.server_variable_dict = {}
        self.client_center_variable_dict = {}

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        input_dim = input_shape[-1]
        self.kernel_regularizer.center = self.add_weight(
            'kernel_center',
            shape=(input_dim, self.units * 4),
            initializer=tf.keras.initializers.constant(0.),
            dtype=self.dtype,
            trainable=False)

        self.recurrent_regularizer.center = self.add_weight(
            'recurrent_kernel_center',
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
                        self.bias_initializer(
                            (self.units * 2,), *args, **kwargs),
                        ])
            else:
                bias_initializer = self.bias_initializer
            self.bias_regularizer.center = self.add_weight(
                'bias_center',
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

        self.client_variable_dict['kernel'] = self.kernel
        self.server_variable_dict['kernel'] = self.kernel
        self.client_center_variable_dict['kernel'] = \
            self.kernel_regularizer.center

        self.client_variable_dict['recurrent_kernel'] = self.recurrent_kernel
        self.server_variable_dict['recurrent_kernel'] = self.recurrent_kernel
        self.client_center_variable_dict['recurrent_kernel'] = \
            self.recurrent_regularizer.center

        if self.use_bias:
            self.client_variable_dict['bias'] = self.bias
            self.server_variable_dict['bias'] = self.bias
            self.client_center_variable_dict['bias'] = \
                self.bias_regularizer.center

        self.built = True


class RNNCentered(tf.keras.layers.RNN):

    def compute_delta(self):
        return self.cell.compute_delta()

    def renew_center(self, center_to_update=True):
        self.cell.renew_center(center_to_update)

    def apply_delta(self, delta):
        self.cell.apply_delta(delta)

    def receive_and_save_weights(self, layer_server):
        self.cell.receive_and_save_weights(layer_server.cell)


class EmbeddingCentered(tf.keras.layers.Embedding, LayerCentered):

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
        super(EmbeddingCentered, self).__init__(
            input_dim, output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=None,
            activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            input_length=input_length,
            **kwargs)

        self.embeddings_regularizer = embeddings_regularizer()
        self.delta_function = tf.subtract
        self.apply_delta_function = tf.add
        self.client_variable_dict = {}
        self.server_variable_dict = {}
        self.client_center_variable_dict = {}

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        def create_weights():
            self.embeddings_regularizer.center = self.add_weight(
                shape=(self.input_dim, self.output_dim),
                name='embeddings_center',
                initializer=tf.keras.initializers.constant(0.),
                dtype=self.dtype,
                trainable=False)

            self.embeddings = self.add_weight(
                shape=(self.input_dim, self.output_dim),
                initializer=self.embeddings_initializer,
                name='embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint)
        if context.executing_eagerly() and context.context().num_gpus():
            with ops.device('cpu:0'):
                create_weights()
        else:
            create_weights()

        self.client_variable_dict['embeddings'] = self.embeddings
        self.server_variable_dict['embeddings'] = self.embeddings
        self.client_center_variable_dict['embeddings'] = \
            self.embeddings_regularizer.center
        self.built = True


class Conv2DCentered(tf.keras.layers.Conv2D, LayerCentered):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
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
        super(Conv2DCentered, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.kernel_regularizer = kernel_regularizer()
        self.bias_regularizer = bias_regularizer()
        self.delta_function = tf.subtract
        self.apply_delta_function = tf.add
        self.client_variable_dict = {}
        self.server_variable_dict = {}
        self.client_center_variable_dict = {}

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        kernel_shape = self.kernel_size + (input_channel, self.filters)

        self.kernel_regularizer.center = \
            self.add_weight('kernel_center',
                            shape=kernel_shape,
                            initializer=tf.keras.initializers.constant(0.),
                            dtype=self.dtype,
                            trainable=False)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias_regularizer.center = \
                self.add_weight('bias_center',
                                shape=(self.filters,),
                                initializer=tf.keras.initializers.constant(0.),
                                dtype=self.dtype,
                                trainable=False)
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})

        self._build_conv_op_input_shape = input_shape
        self._build_input_channel = input_channel
        self._padding_op = self._get_padding_op()
        self._conv_op_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2)
        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=self.kernel.shape,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self._padding_op,
            data_format=self._conv_op_data_format)

        self.client_variable_dict['kernel'] = self.kernel
        self.server_variable_dict['kernel'] = self.kernel
        self.client_center_variable_dict['kernel'] = \
            self.kernel_regularizer.center

        if self.use_bias:
            self.client_variable_dict['bias'] = self.bias
            self.server_variable_dict['bias'] = self.bias
            self.client_center_variable_dict['bias'] = \
                self.bias_regularizer.center

        self.built = True