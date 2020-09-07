from data_utils import federated_dataset, batch_dataset
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from utils import gpu_session
from tensorflow_probability.python.layers.util import default_mean_field_normal_fn
import tensorflow.compat.v1 as tf1
from dense_reparametrization_shared import DenseReparametrizationShared
from tfp_utils import renormalize_mean_field_normal_fn
import time

gpu_session(1)

data_set_conf = {"name": "femnist", "num_clients": 1}

lr = 0.001
BATCH_SIZE = 20
KL_WEIGHT = 1e-6
scale_init = -5

federated_train_data, federated_test_data, train_size, test_size = federated_dataset(data_set_conf)
train_data = federated_train_data[0]
test_data = federated_test_data[0]
train_size = train_size[0]

train_data_batched = batch_dataset(train_data, BATCH_SIZE)
test_data_batched = batch_dataset(test_data, BATCH_SIZE)


class CustomTensorboard(tf.keras.callbacks.TensorBoard):

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        self._log_metrics(logs, prefix='', step=epoch)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_weights(epoch)

        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self._log_embeddings(epoch)


kernel_divergence_fn = (lambda q, p, ignore: KL_WEIGHT * kl_lib.kl_divergence(q, p) / train_size)
#kernel_posterior_fn = renormalize_mean_field_normal_fn
kernel_posterior_fn = default_mean_field_normal_fn(
    untransformed_scale_initializer=tf1.initializers.random_normal(
        mean=scale_init, stddev=0.1))

untransformed_scale_initializer = tf.random_normal_initializer(mean=scale_init, stddev=0.1)
#layer = DenseReparametrizationShared
layer = tfp.layers.DenseReparameterization

param_dict = {#'untransformed_scale_initializer': untransformed_scale_initializer,
              'kernel_posterior_fn': kernel_posterior_fn,
              'kernel_divergence_fn': kernel_divergence_fn
              }

model = tf.keras.Sequential([layer(100, input_shape=[784], activation="relu", **param_dict),
                             layer(100, activation="relu", **param_dict),
                             layer(10, activation="softmax", **param_dict)
                             ])

model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['sparse_categorical_accuracy', tf.keras.metrics.SparseCategoricalAccuracy()],
              #run_eagerly=True
              )

model.fit(train_data_batched, epochs=1000, validation_data=test_data_batched,
          callbacks=[CustomTensorboard(log_dir='logs/test/tfp_' + str(time.time()),  histogram_freq=1)])
