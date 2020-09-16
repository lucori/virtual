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
from tensorflow.python.eager import context
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from natural_raparametrization_layer import DenseReparametrizationNaturalShared, \
                                            DenseLocalReparametrizationNaturalShared
from normal_natural import NormalNatural

gpu_session(1)

data_set_conf = {"name": "femnist", "num_clients": 1}

lr = 100.
BATCH_SIZE = 20
KL_WEIGHT = 0.000001
scale_init = -5
prec_init = 20000.
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


class CustomTensorboard(tf.keras.callbacks.TensorBoard):

    def __init__(self, *args, **kwargs):
        optimizer = kwargs.pop('optimizer', None)
        super(CustomTensorboard, self).__init__(*args, **kwargs)
        self.optimizer = optimizer

    def _log_distr(self, epoch):
        """Logs the weights of the gaussian distributions to TensorBoard."""
        writer = self._get_writer(self._train_run_name)
        with context.eager_mode(), \
             writer.as_default(), \
             summary_ops_v2.always_record_summaries():
            for layer in self.model.layers:
                summary_ops_v2.histogram(layer.name + '/gamma_reparametrized', layer.kernel_posterior.distribution.gamma,
                                         step=epoch)
                summary_ops_v2.histogram(layer.name + '/prec_reparametrized', layer.kernel_posterior.distribution.prec,
                                         step=epoch)
            writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        self._log_metrics(logs, prefix='', step=epoch)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_weights(epoch)
            self._log_distr(epoch)

        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self._log_embeddings(epoch)




kernel_divergence_fn = (lambda q, p, ignore: KL_WEIGHT * kl_lib.kl_divergence(q, p) / train_size)
kernel_posterior_fn = renormalize_mean_field_normal_fn
#kernel_posterior_fn = default_mean_field_normal_fn(
#    untransformed_scale_initializer=tf1.initializers.random_normal(
#        mean=scale_init, stddev=0.1))

untransformed_scale_initializer = tf.random_normal_initializer(mean=scale_init, stddev=0.1)
#layer = DenseReparametrizationShared
#layer = tfp.layers.DenseReparameterization
#layer = DenseReparametrizationNaturalShared
layer = DenseLocalReparametrizationNaturalShared
precision_initializer = tf.random_normal_initializer(mean=prec_init, stddev=1)


param_dict = {'untransformed_scale_initializer': untransformed_scale_initializer,
              #'precision_initializer': precision_initializer,
              #'kernel_posterior_fn': kernel_posterior_fn,
              'kernel_divergence_fn': kernel_divergence_fn,
              'bias_posterior_fn': None
              }

model = tf.keras.Sequential([layer(100, input_shape=[784], activation="relu", **param_dict),
                             layer(100, activation="relu", **param_dict),
                             layer(10, activation="softmax", **param_dict)
                             ])


class MultiSampleEstimator(tf.keras.Model):

    def __init__(self, model, num_samples):
        super(MultiSampleEstimator, self).__init__()
        self.model = model
        self.num_samples = num_samples

    def call(self, inputs, training=None, mask=None):
        output = []
        for _ in range(self.num_samples):
            output.append(self.model.call(inputs, training, mask))
        output = tf.stack(output)
        output = tf.math.reduce_mean(output, axis=0)
        return output

#model = MultiSampleEstimator(model, num_samples=num_samples)
optimizer = tf.keras.optimizers.Adam(lr)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['sparse_categorical_accuracy', tf.keras.metrics.SparseCategoricalAccuracy()],
              run_eagerly=True
              )


model.fit(train_data_batched, epochs=1000, validation_data=test_data_batched,
          callbacks=[CustomTensorboard(log_dir='logs/test/natural_shared_' + str(seed) + '_' + str(time.time()),
                                       histogram_freq=0, profile_batch=0)]
          )
