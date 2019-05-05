import tensorflow as tf
import os
import tensorflow_probability as tfp
import numpy as np
import GPUtil
from tensorflow_probability.python import  distributions as tfd
from tensorflow_probability.python.layers import DenseReparameterization
from server import Server
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import initializers


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


def gpu_session(num_gpus=None, gpus=None):
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    if num_gpus:
        if num_gpus >0:
            os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
            os.environ["CUDA_VISIBLE_DEVICES"] = set_free_gpus(num_gpus)
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"])
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    if gpus or num_gpus>0:
        distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        return config
    else:
        return None


def set_free_gpus(num):
    # num: integer; number of GPUs that shall be allocated
    # returns: string; listing a total of 'num' available GPUs.

    list_gpu = GPUtil.getAvailable(limit=num, maxMemory=0.02)
    return str(list_gpu)[1:-1]


def multivariate_normal_fn(mu, u_sigma):
    def _fn(dtype, shape, name, trainable, add_variable_fn):
        del name, trainable, add_variable_fn, shape
        scale = compute_scale(dtype, u_sigma)
        dist = tfd.Normal(loc=mu, scale=scale)
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    return _fn


def compute_scale(dtype, u_sigma):
    return np.finfo(dtype.as_numpy_dtype).eps + tf.nn.softplus(u_sigma)


def compute_gaussian_ratio(mu1, u_sigma1, mu2, u_sigma2):
    def _fn(dtype, shape, name, trainable, add_variable_fn):
        del name, trainable, add_variable_fn, shape
        scale1 = compute_scale(dtype, u_sigma1)
        scale2 = compute_scale(dtype, u_sigma2)
        scale = tf.math.reciprocal(tf.math.reciprocal(scale1) + tf.math.reciprocal(scale2))
        mu = tf.math.multiply(scale,
                              tf.math.multiply(tf.math.reciprocal(scale1), mu1) +
                              tf.math.multiply(tf.math.reciprocal(scale2), mu2))
        dist = tfd.Normal(loc=mu, scale=scale)
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

    return _fn


def gaussian_ratio_par(v1, v2):
    if v1 and v2:
        mu1 = v1[0]
        u_sigma1 = v1[1]
        mu2 = v2[0]
        u_sigma2 = v2[1]
        scale1 = tf.nn.softplus(u_sigma1)
        scale2 = tf.nn.softplus(u_sigma2)
        scale = tf.math.reciprocal(tf.math.reciprocal(scale1) + tf.math.reciprocal(scale2))
        mu = tf.math.multiply(scale,
                              tf.math.multiply(tf.math.reciprocal(scale1), mu1) +
                              tf.math.multiply(tf.math.reciprocal(scale2), mu2))
        return mu, tf.math.log(tf.math.exp(scale)-1)
    else:
        return []


def get_refined_prior(l1, w2):
    w1 = l1.get_weights()
    if w2:
        return compute_gaussian_ratio(w1[0], w1[1], w2[0], w2[1])
    else:
        return get_posterior_from_layer(l1)


def get_posterior_from_layer(l):
    if l.get_weights():
        return multivariate_normal_fn(l.get_weights()[0], l.get_weights()[1])
    else:
        return []


def clone(layer, data_set_size=None, n_samples=None, **kwargs):
    config = layer.get_config()
    for key, value in kwargs.items():
        if key == 'name':
            config[key] = config[key] + value
        else:
            config[key] = value
    if issubclass(layer.__class__, DenseReparameterization):
        sub_config = dict((k, config[k]) for k in ('units', 'activation', 'name', 'activity_regularizer', 'trainable') if k in config)
        keys_config = set(sub_config.keys())
        kwargs.pop('name', None)
        keys_kwargs = set(kwargs.keys())
        union = keys_config | keys_kwargs
        args = {}
        for key in union:
            if key in kwargs:
                args[key] = kwargs[key]
            else:
                args[key] = sub_config[key]
        args['kernel_divergence_fn'] = lambda q, p, _: tfp.distributions.kl_divergence(q, p)/(data_set_size*n_samples)
        return layer.__class__(**args)
    else:
        return layer.__class__.from_config(config)


def get_mlp_server(input_shape, layer, layer_units, activations, data_set_size, num_samples):
    server = Server()
    server.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    server.add(tf.keras.layers.Flatten())
    for i, (u, act) in enumerate(zip(layer_units, activations)):
        server.add(layer(u, activation=act, name='lateral' + str(i),
              kernel_divergence_fn=
              lambda q, p, _: tfp.distributions.kl_divergence(q, p)/(data_set_size*num_samples)))
    return server


def aulc(virtual_history, quantity='categorical_accuracy'):
    history_first_pass = [v[0] for v in virtual_history.values()]
    quantity_per_task = [h[quantity] for h in history_first_pass]
    epochs = [len(q) for q in quantity_per_task]
    min_epochs = min(epochs)
    aulcs = [np.trapz(q[:min_epochs]) for q in quantity_per_task]

    return np.array(aulcs)
