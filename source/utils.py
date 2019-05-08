import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow_probability.python import  distributions as tfd
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import initializers
from general_utils import clone


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


class DenseReparameterizationPriorUpdate(tfp.layers.DenseReparameterization):

    def update_prior(self, kernel_prior_fn):
        input_shape = self.input_shape
        in_size = input_shape[-1]
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        self.kernel_prior = kernel_prior_fn(dtype, [in_size, self.units], 'kernel_prior',
                                            self.trainable, self.add_variable)
        self._losses = []
        self._apply_divergence(self.kernel_divergence_fn,
                               self.kernel_posterior,
                               self.kernel_prior,
                               self.kernel_posterior_tensor,
                               name='divergence_kernel')


class LateralConnection(tf.keras.layers.Layer):

    def __init__(self, layer, data_set_size, n_samples, **kwargs):
        super(LateralConnection, self).__init__(**kwargs)
        self.layer = layer
        self.data_set_size = data_set_size
        self.n_samples = n_samples
        self.from_client = clone(self.layer, data_set_size=self.data_set_size, n_samples=self.n_samples,
                     activation='linear', name=self.name + '_from_client')
        self.from_server = clone(self.layer, data_set_size=self.data_set_size, n_samples=self.n_samples,
                     name=self.name + '_from_server1')
        self.from_server_projection = clone(self.layer, data_set_size=self.data_set_size, n_samples=self.n_samples,
                     activation='linear', name=self.name + '_from_server2')
        self.gate = Gate()
        self.layers = [self.from_client, self.from_server, self.from_server_projection, self.gate]

    def call(self, inputs):
        out1 = self.from_client(inputs[0])
        out2 = self.from_server(inputs[1])
        out2 = self.from_server_projection(out2)
        out2 = self.gate(out2)
        outputs = tf.keras.layers.add([out1, out2], name=self.layer.name + '_add' + self.name)
        outputs = tf.keras.layers.Activation(self.layer.get_config()['activation'],
                                       name=self.layer.name + '_activation' + self.name)(outputs)
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


def multivariate_normal_fn(mu, u_sigma):
    def _fn(dtype, shape, name, trainable, add_variable_fn):
        del name, trainable, add_variable_fn, shape
        scale = compute_scale(u_sigma)
        dist = tfd.Normal(loc=mu, scale=scale)
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    return _fn


def compute_scale(u_sigma):
    dtype = u_sigma.dtype
    if hasattr(dtype, 'as_numpy_dtype'):
        dtype = dtype.as_numpy_dtype
    return np.finfo(dtype).eps + softplus(u_sigma)


def compute_gaussian_ratio(mu1, u_sigma1, mu2, u_sigma2):
    def _fn(dtype, shape, name, trainable, add_variable_fn):
        del name, trainable, add_variable_fn, shape
        mu, scale = _gaussian_ratio_par(mu1, u_sigma1, mu2, u_sigma2)
        dist = tfd.Normal(loc=mu, scale=scale)
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    return _fn


def softplus(x):
    return np.log(np.exp(x) + 1)


def softminus(x):
    return np.log(np.exp(x) - 1)


def softminus_tensor(x):
    return tf.math.log(tf.math.exp(x) -1)


def gaussian_ratio_par(v1, v2):
    if v1 and v2:
        mu1 = v1[0]
        u_sigma1 = v1[1]
        mu2 = v2[0]
        u_sigma2 = v2[1]
        mu, scale = _gaussian_ratio_par(mu1, u_sigma1, mu2, u_sigma2)
        return mu, softminus(scale)
    else:
        return []


def _gaussian_ratio_par(mu1, u_sigma1, mu2, u_sigma2):
    scale1 = compute_scale(u_sigma1)
    scale2 = compute_scale(u_sigma2)
    #scale1 = tf.debugging.check_numerics(scale1, 'error1')
    #scale2 = tf.debugging.check_numerics(scale2, 'error2')
    #with tf.control_dependencies([tf.assert_none_equal(scale1, scale2)]):
    #scale = tf.math.reciprocal(np.finfo(scale1.dtype.as_numpy_dtype).eps + tf.math.reciprocal(scale1) -
    #                           tf.math.reciprocal(scale2))
    #scale = tf.debugging.check_numerics(scale, 'error')
    #mu = tf.math.multiply(scale,
    #                      tf.math.multiply(tf.math.reciprocal(scale1), mu1) -
    #                      tf.math.multiply(tf.math.reciprocal(scale2), mu2))
    print(scale1,scale2)
    print(np.any(scale1>=scale2))
    scale = 1/(np.finfo(scale1.dtype).eps + 1/scale1 - 1/scale2)
    mu = scale*(1/scale1*mu1 - 1/scale2*mu2)
    return mu, scale


def get_refined_prior(l1, w2):
    w1 = l1.get_weights()
    if w2:
        return compute_gaussian_ratio(w1[0], w1[1], w2[0], w2[1])
    else:
        return get_posterior_from_layer(l1)


def get_posterior_from_layer(l):
    weights = l.get_weights()
    if len(weights) > 1:
        return multivariate_normal_fn(weights[0], weights[1])
    else:
        return


def prior_wrapper(posterior, l):
    if hasattr(l, 'layers'):
        return [posterior(la) for la in l.layers]
    else:
        return posterior(l)
