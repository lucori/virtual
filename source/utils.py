import tensorflow as tf
import os
import tensorflow_probability as tfp
import numpy as np
import GPUtil
from tensorflow_probability.python import  distributions as tfd
from tensorflow_probability.python.layers import DenseReparameterization


def mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = tf.keras.utils.to_categorical(y_train.astype('int32'))
    y_test = tf.keras.utils.to_categorical(y_test.astype('int32'))
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    x_train /= 126
    x_test /= 126
    return x_train, y_train, x_test, y_test


def permuted_mnist(x_train, x_test):

    def shuffle(x, indx):
        shape = x[0].shape
        for i, _ in enumerate(x):
            x[i] = (x[i].flatten()[indx]).reshape(shape)
        return x

    indx = np.random.permutation(x_train[0].size)
    x_train = shuffle(x_train, indx)
    x_test = shuffle(x_test, indx)
    return x_train, x_test


def build_input_pipeline(x, y, batch_size, iterator=False):
    training_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    training_dataset = training_dataset.repeat().batch(batch_size)
    if not iterator:
        return training_dataset
    else:
        training_iterator = training_dataset.make_one_shot_iterator()
        images, labels = training_iterator.get_next()

        return images, labels, training_iterator


def gpu_session(num_gpus):
    if num_gpus > 0:
        os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = set_free_gpus(num_gpus)
        print(os.environ["CUDA_VISIBLE_DEVICES"])
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


def sparse_array(position, size):
    array = [0]*size
    array[position] = 1
    return array


def permuted_mnist_for_n_tasks(num_tasks):
    x_train, y_train, x_test, y_test = mnist_data()
    x = []
    x_t = []
    y = []
    y_t = []
    for _ in range(num_tasks):
        x_train_perm, x_test_perm = permuted_mnist(x_train, x_test)
        x.append(x_train_perm)
        x_t.append(x_test_perm)
        y.append(y_train)
        y_t.append(y_test)

    return x, y, x_t, y_t