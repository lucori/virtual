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

    def build(self, input_shape):
        super(DenseReparameterizationPriorUpdate, self).build(input_shape)
        self.reparametrized = False

    def update_prior(self, kernel_prior_fn):
        input_shape = self.input_shape
        in_size = input_shape[-1]
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        self.kernel_prior = kernel_prior_fn(dtype, [in_size, self.units], 'kernel_prior',
                                            self.trainable, self.add_variable)
        self.reinitialize_weights()
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

    def reinitialize_weights(self):
        tf.assign(self.weights[0], tf.random_normal_initializer(stddev=0.1)(self.weights[0].shape))
        tf.assign(self.weights[1], tf.random_normal_initializer(mean=-3.0, stddev=0.1)(self.weights[1].shape))
        tf.assign(self.weights[2], tf.random_normal_initializer(stddev=0.1)(self.weights[2].shape))

    def get_weights(self):
        weights = super(DenseReparameterizationPriorUpdate, self).get_weights()
        if self.reparametrized:
            weights[0], weights[1] = gaussian_prod_par(weights, [self.prior_loc, softminus(self.prior_scale)])
        return weights

    def get_t(self):
        return super(DenseReparameterizationPriorUpdate, self).get_weights()


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
    return np.log(np.exp(x) + 1.)


def softminus(x):
    return np.log(np.exp(x) - 1.)


def gaussian_ratio_par(v1, v2):
    mu1 = v1[0]
    u_sigma1 = v1[1]
    mu2 = v2[0]
    u_sigma2 = v2[1]
    mu, scale = _gaussian_ratio_par(mu1, u_sigma1, mu2, u_sigma2)
    return mu, softminus(scale)


def gaussian_prod_par(v1, v2):
    mu1 = v1[0]
    u_sigma1 = v1[1]
    mu2 = v2[0]
    u_sigma2 = v2[1]
    mu, scale = _gaussian_prod_par(mu1, u_sigma1, mu2, u_sigma2)
    return mu, softminus(scale)


def _gaussian_ratio_par(mu1, u_sigma1, mu2, u_sigma2):
    scale1 = compute_scale(u_sigma1)
    scale2 = compute_scale(u_sigma2)
    scale = np.sqrt(1 / (1 / scale1 ** 2 - 1 / scale2 ** 2))
    mu = scale ** 2 * (1 / scale1 ** 2 * mu1 - 1 / scale2 ** 2 * mu2)
    return mu, scale


def _gaussian_prod_par(mu1, u_sigma1, mu2, u_sigma2):
    scale1 = compute_scale(u_sigma1)
    scale2 = compute_scale(u_sigma2)
    scale = np.sqrt(1 / (1 / scale1 ** 2 + 1 / scale2 ** 2))
    mu = scale ** 2 * (1 / scale1 ** 2 * mu1 + 1 / scale2 ** 2 * mu2)
    return mu, scale


def get_refined_prior(l1, w2):
    w1 = l1.get_weights()
    return compute_gaussian_ratio(w1[0], w1[1], w2[0], w2[1])


def get_posterior_from_layer(l):
    weights = l.get_weights()
    return multivariate_normal_fn(weights[0], weights[1])


def prior_wrapper(posterior, l):
    if hasattr(l, 'layers'):
        return [posterior(la) for la in l.layers]
    else:
        return posterior(l)
