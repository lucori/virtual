import tensorflow as tf
from data_utils import post_process_datasets, federated_dataset
from fed_prox import FedProx
import tensorflow_federated as tff
from centered_l2_regularizer import DenseCentered, CenteredL2Regularizer, LSTMCellCentered, RNNCentered, \
    EmbeddingCentered
from utils import gpu_session
import os

SEQ_LENGTH = 80
BATCH_SIZE = 10
BUFFER_SIZE = 10000
embedding_dim = 8
rnn_units = 256
VOCAB_SIZE = 86
num_clients = 100
num_rounds = 100
clients_per_round = 5
epochs_per_round = 2
learning_rate = 0.1

#gpu_session(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=vocab, values=tf.constant(list(range(1, len(vocab)+1)),
                                       dtype=tf.int64)),
    default_value=tf.cast(0, tf.int64))


def to_ids(x):
    s = tf.reshape(x['snippets'], shape=[1])
    chars = tf.strings.bytes_split(s).values
    ids = table.lookup(chars)
    return ids


def split_input_target(chunk):
    input_text = tf.map_fn(lambda x: x[:-1], chunk)
    target_text = tf.map_fn(lambda x: x[1:], chunk)
    return (input_text, target_text)


def preprocess(dataset):
    return (
        # Map ASCII chars to int64 indexes using the vocab
        dataset.map(to_ids)
            # Split into individual chars
            .unbatch())
            # Form example sequences of SEQ_LENGTH +1


def postprocess(dataset):
    return (dataset.batch(SEQ_LENGTH + 1, drop_remainder=False)
                    # Shuffle and form minibatches
                    .shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=[SEQ_LENGTH + 1], drop_remainder=False,
                                                      padding_values=tf.cast(0, tf.int64))
                    .padded_batch(1, padded_shapes=[BATCH_SIZE, SEQ_LENGTH + 1], padding_values=tf.cast(0, tf.int64),
                                  drop_remainder=False).unbatch()
                    # And finally split into (input, target) tuples,
                    # each of length SEQ_LENGTH.
                    .map(split_input_target))


train_data, test_data = tff.simulation.datasets.shakespeare.load_data()


def data(client, source):
    return postprocess(preprocess(source.create_tf_dataset_for_client(client)))


clients = train_data.client_ids[0:num_clients]


train_size = [len(list(preprocess(train_data.create_tf_dataset_for_client(client)))) for client in clients]
test_size = [len(list(preprocess(test_data.create_tf_dataset_for_client(client)))) for client in clients]
print(train_size)
print(test_size)
clients, train_size, test_size = zip(*[(client, train_size[i], test_size[i])
                                     for i, client in enumerate(clients) if train_size[i] > 0 and test_size[i] > 0])
num_clients = len(clients)
federated_train_data = [data(client, train_data) for client in clients]
federated_test_data = [data(client, test_data) for client in clients]


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


def model_fn(model_class, num_clients=1):
    keras_model = model_class([
        EmbeddingCentered(VOCAB_SIZE+1, embedding_dim,
                          batch_input_shape=[BATCH_SIZE, SEQ_LENGTH],
                          embeddings_regularizer=lambda:CenteredL2Regularizer(0.),
                          mask_zero=True),
        RNNCentered(LSTMCellCentered(rnn_units,
                                             recurrent_initializer='glorot_uniform',
                                             kernel_regularizer=lambda:CenteredL2Regularizer(0.),
                                             recurrent_regularizer=lambda:CenteredL2Regularizer(0.),
                                             bias_regularizer=lambda:CenteredL2Regularizer(0.)),
                            return_sequences=True,
                            stateful=True),
        RNNCentered(LSTMCellCentered(rnn_units,
                                             recurrent_initializer='glorot_uniform',
                                             kernel_regularizer=lambda:CenteredL2Regularizer(0.),
                                             recurrent_regularizer=lambda:CenteredL2Regularizer(0.),
                                             bias_regularizer=lambda:CenteredL2Regularizer(0.)),
                            return_sequences=True,
                            stateful=True),
        DenseCentered(VOCAB_SIZE+1, kernel_regularizer=lambda:CenteredL2Regularizer(0.),
                      bias_regularizer=lambda:CenteredL2Regularizer(0.))
    ])
    return compile(keras_model)



logdir = 'logs/LSTM'


fed_prox_process = FedProx(model_fn, num_clients)
fed_prox_process.fit(federated_train_data, num_rounds, clients_per_round,
                     epochs_per_round, train_size=train_size, test_size=test_size,
                     federated_test_data=federated_test_data,
                     tensorboard_updates=1,
                     logdir=logdir)
