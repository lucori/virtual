import tensorflow as tf
import os
import tensorflow_probability as tfp
import numpy as np
import GPUtil
from tensorflow_probability.python.layers import DenseReparameterization
from server import Server


def gpu_session(num_gpus=None, gpus=None):
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    elif num_gpus:
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