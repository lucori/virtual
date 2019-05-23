import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow_probability.python import distributions as tfd
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import initializers
from general_utils import clone
from utils import *
from tensorflow_probability.python.layers import util as tfp_layers_util


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


class DenseReparameterizationServer(tfp.layers.DenseReparameterization):

    def __init__(self, units,
                 activation=None,
                 activity_regularizer=None,
                 trainable=True,
                 kernel_posterior_fn=renormalize_mean_field_normal_fn,
                 kernel_posterior_tensor_fn=(lambda d: d.sample()),
                 kernel_prior_fn=default_tensor_multivariate_normal_fn,
                 kernel_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p)),
                 bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(is_singular=True),
                 bias_posterior_tensor_fn=(lambda d: d.sample()),
                 bias_prior_fn=None,
                 bias_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p)),
                 **kwargs
                 ):

        super(DenseReparameterizationServer, self).__init__(units,
                                                                 activation=activation,
                                                                 activity_regularizer=activity_regularizer,
                                                                 trainable=trainable,
                                                                 kernel_posterior_fn=kernel_posterior_fn,
                                                                 kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
                                                                 kernel_prior_fn=kernel_prior_fn,
                                                                 kernel_divergence_fn=kernel_divergence_fn,
                                                                 bias_posterior_fn=bias_posterior_fn,
                                                                 bias_posterior_tensor_fn=bias_posterior_tensor_fn,
                                                                 bias_prior_fn=bias_prior_fn,
                                                                 bias_divergence_fn=bias_divergence_fn,
                                                                 **kwargs)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        in_size = input_shape.with_rank_at_least(2)[-1].value
        if in_size is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self._input_spec = tf.layers.InputSpec(min_ndim=2, axes={-1: in_size})

        # If self.dtype is None, build weights using the default dtype.
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

        if self.kernel_prior_fn is None:
            self.kernel_prior = None
        else:
            self.kernel_prior = self.kernel_prior_fn(
                dtype, [in_size, self.units], 'kernel_prior',
                self.trainable, self.add_variable)
        self._built_kernel_divergence = False

        self.kernel_posterior = self.kernel_posterior_fn(self.non_trainable_variables)(
            dtype, [in_size, self.units], 'kernel_posterior',
            self.trainable, self.add_variable)

        if self.bias_posterior_fn is None:
            self.bias_posterior = None
        else:
            self.bias_posterior = self.bias_posterior_fn(
                dtype, [self.units], 'bias_posterior',
                self.trainable, self.add_variable)

        if self.bias_prior_fn is None:
            self.bias_prior = None
        else:
            self.bias_prior = self.bias_prior_fn(
                dtype, [self.units], 'bias_prior',
                self.trainable, self.add_variable)
        self._built_bias_divergence = False
        self.data_set_size = tf.Variable(1, trainable=False)
        self.built = True

    def _apply_divergence(self, divergence_fn, posterior, prior,
                          posterior_tensor, name):
        if(divergence_fn is None or
                posterior is None or
                prior is None):
            divergence = None
            return
        divergence = tf.identity(
            divergence_fn(
                posterior, prior, posterior_tensor),
            name=name)
        divergence = tf.math.divide(divergence, tf.cast(tf.identity(self.data_set_size), tf.float32))
        self.add_loss(divergence)

    def update_prior(self, new_prior_tensors):
        tf.keras.backend.set_value(self.non_trainable_variables[0], new_prior_tensors[0])
        tf.keras.backend.set_value(self.non_trainable_variables[1], new_prior_tensors[1])

    def get_weights(self):
        weights = super(DenseReparameterizationServer, self).get_weights()
        return gaussian_prod_par(weights[2:], weights[:2])

    def get_t(self):
        return super(DenseReparameterizationServer, self).get_weights()[2:]

    def set_data_set_size(self, size):
        tf.keras.backend.set_value(self.data_set_size, size)


class LateralConnection(tf.keras.layers.Layer):

    def __init__(self, layer, data_set_size, n_samples, **kwargs):
        super(LateralConnection, self).__init__(**kwargs)
        self.layer_name = layer.name
        self.layer_conf = layer.get_config()
        self.data_set_size = data_set_size
        self.n_samples = n_samples
        self.from_client = clone(layer, data_set_size=self.data_set_size, n_samples=self.n_samples,
                     activation='linear', name=self.name + '_from_client')
        self.from_server = clone(layer, data_set_size=self.data_set_size, n_samples=self.n_samples,
                     name=self.name + '_from_server1')
        self.from_server_projection = clone(layer, data_set_size=self.data_set_size, n_samples=self.n_samples,
                     activation='linear', name=self.name + '_from_server2')
        self.gate = Gate()
        self.layers = [self.from_client, self.from_server, self.from_server_projection, self.gate]

    def call(self, inputs):
        out1 = self.from_client(inputs[0])
        out2 = self.from_server(inputs[1])
        out2 = self.from_server_projection(out2)
        out2 = self.gate(out2)
        outputs = tf.keras.layers.add([out1, out2], name=self.layer_name + '_add' + self.name)
        outputs = tf.keras.layers.Activation(self.layer_conf['activation'],
                                       name=self.layer_name + '_activation' + self.name)(outputs)
        return outputs

    @property
    def weights(self):
        weights = self.from_client.weights
        weights.extend(self.from_server.weights)
        weights.extend(self.from_server_projection.weights)
        weights.extend(self.gate.weights)
        return weights

    def update_prior(self, kernel_prior_fn):
        layers_to_update = [l for l in self.layers if hasattr(l, 'update_prior')]
        for l, f in zip(layers_to_update, kernel_prior_fn):
            l.update_prior(f)


class DenseFlipOutPriorUpdate(tfp.layers.DenseFlipout):

    def __init__(self, units,
                 activation=None,
                 activity_regularizer=None,
                 trainable=True,
                 kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
                 kernel_posterior_tensor_fn=(lambda d: d.sample()),
                 kernel_prior_fn=default_np_multivariate_normal_fn,
                 kernel_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p)),
                 bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(is_singular=True),
                 bias_posterior_tensor_fn=(lambda d: d.sample()),
                 bias_prior_fn=None,
                 bias_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p)),
                 **kwargs
                 ):

        super(DenseFlipOutPriorUpdate, self).__init__(units,
                                                                 activation=activation,
                                                                 activity_regularizer=activity_regularizer,
                                                                 trainable=trainable,
                                                                 kernel_posterior_fn=kernel_posterior_fn,
                                                                 kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
                                                                 kernel_prior_fn=kernel_prior_fn,
                                                                 kernel_divergence_fn=kernel_divergence_fn,
                                                                 bias_posterior_fn=bias_posterior_fn,
                                                                 bias_posterior_tensor_fn=bias_posterior_tensor_fn,
                                                                 bias_prior_fn=bias_prior_fn,
                                                                 bias_divergence_fn=bias_divergence_fn,
                                                                 **kwargs)

    def build(self, input_shape):
        super(DenseFlipOutPriorUpdate, self).build(input_shape)
        self.reparametrized = False

    def update_divergence(self, size):
        self.kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p)/size

    def update_prior(self, kernel_prior_fn):
        input_shape = self.input_shape
        in_size = input_shape[-1]
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        self.kernel_prior = kernel_prior_fn(dtype, [in_size, self.units], 'kernel_prior',
                                            self.trainable, self.add_variable)
        if self.reparametrized:
            self.reparametrize_posterior()
        self.update_loss()

    def update_loss(self):
        self._losses = []
        self._apply_divergence(self.kernel_divergence_fn,
                               self.kernel_posterior,
                               self.kernel_prior,
                               self.kernel_posterior_tensor,
                               name='divergence_kernel')

    def reparametrize_posterior(self):
        prior_par = self.kernel_prior.parameters
        self.prior_loc = prior_par['distribution'].parameters['loc']
        self.prior_scale = prior_par['distribution'].parameters['scale']
        self.kernel_posterior = self.compute_new_posterior()
        self.update_loss()
        self.reparametrized = True

    def compute_new_posterior(self):
        scale1 = (np.finfo(self.weights[1].dtype.as_numpy_dtype).eps +
                  tf.nn.softplus(self.weights[1]))
        scale = tf.math.sqrt(tf.math.reciprocal(tf.math.reciprocal(tf.math.square(scale1)) +
                                                tf.math.reciprocal(tf.math.square(self.prior_scale))))
        loc = tf.math.multiply(tf.math.square(scale), tf.math.multiply(tf.math.reciprocal(tf.math.square(scale1)),
                                                                       self.weights[0]) +
                               tf.math.multiply(tf.math.reciprocal(tf.math.square(self.prior_scale)), self.prior_loc))
        dist = tfd.Normal(loc=loc, scale=scale)
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    def get_weights(self):
        weights = super(DenseFlipOutPriorUpdate, self).get_weights()
        if self.reparametrized:
            weights[0], weights[1] = gaussian_prod_par(weights, [self.prior_loc, softminus(self.prior_scale)])
        return weights

    def get_t(self):
        return super(DenseFlipOutPriorUpdate, self).get_weights()