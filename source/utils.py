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
                                                                 maxval=0.01),
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


def get_mlp_server(input_shape, layer, layer_units, activations, data_set_size, num_samples):
    server = Server()
    server.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    server.add(tf.keras.layers.Flatten())
    for i, (u, act) in enumerate(zip(layer_units, activations)):
        server.add(layer(u, activation=act, name='lateral' + str(i),
              kernel_divergence_fn=
              lambda q, p, _: tfp.distributions.kl_divergence(q, p)/(data_set_size*num_samples)))
    return server


def femnist_data(num_tasks, global_data=False, train_set_size_per_user=-1, test_set_size_per_user=-1):
    import os
    import json
    data = []
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
    dir_path_train = os.path.join(dir_path, 'leaf/data/femnist/data/train')
    dir_path_test = os.path.join(dir_path, 'leaf/data/femnist/data/test')

    for file_name in os.listdir(dir_path_train):
        file = os.path.join(dir_path_train, file_name)
        file = open(file)
        data.append(json.load(file))
        file.close()
    data_merge_train = {'users': [], 'num_samples': [], 'user_data': {}}
    for d in data:
        data_merge_train['users'] = data_merge_train['users'] + d['users']
        data_merge_train['num_samples']= data_merge_train['num_samples'] + d['num_samples']
        data_merge_train['user_data'].update(d['user_data'])

    data = []
    for file_name in os.listdir(dir_path_test):
        file = os.path.join(dir_path_test, file_name)
        file = open(file)
        data.append(json.load(file))
        file.close()
    data_merge_test = {'users': [], 'num_samples': [], 'user_data': {}}
    for d in data:
        data_merge_test['users'] = data_merge_test['users'] + d['users']
        data_merge_test['num_samples'] = data_merge_test['num_samples'] + d['num_samples']
        data_merge_test['user_data'].update(d['user_data'])

    x = []
    y = []
    xt = []
    yt = []

    users = [u for u, n in zip(data_merge_train['users'], data_merge_train['num_samples']) if n >= train_set_size_per_user]
    users = users[0:num_tasks]
    for user in users:
        x.append(np.array(data_merge_train['user_data'][user]['x'])[0:train_set_size_per_user])
        y.append(tf.keras.utils.to_categorical(np.array(data_merge_train['user_data'][user]['y']),
                                               num_classes=62)[0:train_set_size_per_user])
        xt.append(np.array(data_merge_test['user_data'][user]['x'])[0:test_set_size_per_user])
        yt.append(tf.keras.utils.to_categorical(np.array(data_merge_test['user_data'][user]['y']),
                                                num_classes=62)[0:test_set_size_per_user])
    if global_data:
        x = np.concatenate(x)
        y = np.concatenate(y)
        xt = np.concatenate(xt)
        yt = np.concatenate(yt)

    return x, y, xt, yt


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
