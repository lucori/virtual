import tensorflow as tf
import numpy as np
from tensorflow_probability.python import distributions as tfd


def default_np_multivariate_normal_fn(dtype, shape, name, trainable, add_variable_fn):
    del name, trainable, add_variable_fn
    dist = tfd.Normal(loc=np.zeros(shape, dtype.as_numpy_dtype), scale=dtype.as_numpy_dtype(1))
    batch_ndims = tf.size(input=dist.batch_shape_tensor())
    return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)


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


def get_refined_prior(w1, w2):
    return compute_gaussian_ratio(w1[0], w1[1], w2[0], w2[1])
