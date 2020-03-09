import tensorflow as tf
from data_utils import federated_dataset, batch_dataset
from centered_layers import DenseCentered, CenteredL2Regularizer, LSTMCellCentered, RNNCentered, \
    EmbeddingCentered
from dense_reparametrization_shared import LSTMCellVariational, DenseReparametrizationShared, RNNVarReparametrized
from utils import gpu_session
from virtual_process import VirtualFedProcess
import os
import tensorflow_probability as tfp
tfd = tfp.distributions


embedding_dim = 8
rnn_units = 256
VOCAB_SIZE = 86
num_clients = 5
num_rounds = 1000
clients_per_round = 3
epochs_per_round = 1
learning_rate = 0.001
BATCH_SIZE = 10
SEQ_LENGTH = 80

#gpu_session(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


federated_train_data, federated_test_data, train_size, test_size = federated_dataset(name='shakespeare',
                                                                                     num_clients=num_clients)

federated_train_data_batched = [batch_dataset(data, BATCH_SIZE, padding=True) for data in federated_train_data]
federated_test_data_batched = [batch_dataset(data, BATCH_SIZE, padding=True) for data in federated_test_data]


class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

    def __init__(self, name='accuracy', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1, 1])
        y_pred = tf.reshape(y_pred, [-1, VOCAB_SIZE+1, 1])
        if sample_weight is not None:
            sample_weight = tf.reshape(sample_weight, [-1, 1])
        return super().update_state(
            y_true, y_pred, sample_weight)


def compile(keras_model):
    keras_model.compile(
      optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[FlattenedCategoricalAccuracy()])
    return keras_model


def model_fn(model_class, train_size=1):
    keras_model = model_class([
        EmbeddingCentered(VOCAB_SIZE+1, embedding_dim,
                          batch_input_shape=[BATCH_SIZE, SEQ_LENGTH],
                          embeddings_regularizer=lambda:CenteredL2Regularizer(0.),
                          mask_zero=True),
        RNNVarReparametrized(LSTMCellVariational(rnn_units,
                                                 recurrent_initializer='glorot_uniform',
                                                 num_clients=num_clients,
                                                 kernel_divergence_fn=(lambda q, p, ignore:
                                                                       tfd.kl_divergence(q, p)/train_size),
                                                 recurrent_kernel_divergence_fn=(lambda q, p, ignore:
                                                                                 tfd.kl_divergence(q, p) / train_size),
                                                 ),
                             return_sequences=True,
                             stateful=True),
        RNNVarReparametrized(LSTMCellVariational(rnn_units,
                                              recurrent_initializer='glorot_uniform',
                                              num_clients=num_clients,
                                              kernel_divergence_fn=(lambda q, p, ignore:
                                                                    tfd.kl_divergence(q, p) / train_size),
                                              recurrent_kernel_divergence_fn=(lambda q, p, ignore:
                                                                              tfd.kl_divergence(q, p) / train_size),
                                              ),
                         return_sequences=True,
                         stateful=True),
        DenseReparametrizationShared(VOCAB_SIZE+1,
                                     num_clients=num_clients,
                                     kernel_divergence_fn=(lambda q, p, ignore:
                                                           tfd.kl_divergence(q, p) / train_size)
                                     )
    ])
    return compile(keras_model)

#TODO: nlp model in experiment_utils function
logdir = 'logs/LSTM'

fed_prox_process = VirtualFedProcess(model_fn, num_clients)
fed_prox_process.fit(federated_train_data_batched, num_rounds, clients_per_round,
                     epochs_per_round, train_size=train_size, test_size=test_size,
                     federated_test_data=federated_test_data_batched,
                     tensorboard_updates=1,
                     logdir=logdir)
