from source.data_utils import federated_dataset, batch_dataset
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from source.utils import gpu_session
from tensorflow_probability.python.layers.util import default_mean_field_normal_fn
import tensorflow.compat.v1 as tf1
from source.dense_reparametrization_shared import DenseReparametrizationShared
from source.tfp_utils import renormalize_mean_field_normal_fn
import time
from tensorflow.python.eager import context
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from source.natural_raparametrization_layer import DenseReparametrizationNaturalShared, \
                                            DenseLocalReparametrizationNaturalShared, \
                                            natural_mean_field_normal_fn, \
                                            natural_tensor_multivariate_normal_fn, \
                                            natural_initializer_fn
from source.normal_natural import NormalNatural
from source.federated_devices import ClientVirtualSequential
from source.utils import CustomTensorboard

gpu_session(1)

data_set_conf = {"name": "femnist", "num_clients": 1}

lr = 5e10
BATCH_SIZE = 20
KL_WEIGHT = 0.0
scale_init = -5
seed = 0
num_samples = 10

tf.random.set_seed(seed)

federated_train_data, federated_test_data, train_size, test_size = \
    federated_dataset(data_set_conf)
train_data = federated_train_data[0]
test_data = federated_test_data[0]
train_size = train_size[0]

train_data_batched = batch_dataset(train_data, BATCH_SIZE)
test_data_batched = batch_dataset(test_data, BATCH_SIZE)

#tf.debugging.enable_check_numerics(stack_height_limit=30, path_length_limit=50)


kernel_divergence_fn = (lambda q, p, ignore: KL_WEIGHT * kl_lib.kl_divergence(q, p) / train_size)

#kernel_posterior_fn = renormalize_mean_field_normal_fn
#kernel_posterior_fn = default_mean_field_normal_fn(
#    untransformed_scale_initializer=tf1.initializers.random_normal(
#        mean=scale_init, stddev=0.1))

kernel_posterior_fn = natural_mean_field_normal_fn
kernel_prior_fn = natural_tensor_multivariate_normal_fn

untransformed_scale_initializer = tf.random_normal_initializer(mean=scale_init, stddev=0.1)
natural_initializer = natural_initializer_fn(untransformed_scale_initializer=untransformed_scale_initializer)

#layer = DenseReparametrizationShared
#layer = tfp.layers.DenseReparameterization
#layer = DenseReparametrizationNaturalShared
layer = DenseLocalReparametrizationNaturalShared
#layer = tf.keras.layers.Dense
#precision_initializer = tf.random_normal_initializer(mean=prec_init, stddev=1)


param_dict = {'untransformed_scale_initializer': untransformed_scale_initializer,
              #'precision_initializer': precision_initializer,
              #'kernel_posterior_fn': kernel_posterior_fn(natural_initializer),
              #'kernel_prior_fn': kernel_prior_fn(),
              'kernel_divergence_fn': kernel_divergence_fn,
              'bias_posterior_fn': None,
              'client_weight': 0.01
              #'use_bias': False,
              }

model = ClientVirtualSequential([layer(100, input_shape=[784], activation="relu", **param_dict),
                          layer(100, activation="relu", **param_dict),
                          layer(10, activation="softmax", **param_dict)
                          ])

model2 = ClientVirtualSequential([layer(100, input_shape=[784], activation="relu", **param_dict),
                          layer(100, activation="relu", **param_dict),
                          layer(10, activation="softmax", **param_dict)
                          ])

optimizer = tf.keras.optimizers.SGD(lr)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['sparse_categorical_accuracy', tf.keras.metrics.SparseCategoricalAccuracy()],
              run_eagerly=True
              )

for layer, layer2 in zip(model.layers, model2.layers):
    for key in layer.server_variable_dict.keys():
        layer.server_variable_dict[key].assign(layer2.client_variable_dict[key])

model.fit(train_data_batched, epochs=1000, validation_data=test_data_batched,
          callbacks=[CustomTensorboard(log_dir='logs/femnist_virtual_100_10/' + str(seed) + '_' + str(time.time()),
                                       histogram_freq=1, profile_batch=0)]
          )
