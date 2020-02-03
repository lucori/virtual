import tensorflow as tf
from dense_reparametrization_shared import DenseReparametrizationShared
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from virtual_process import VirtualFedProcess
import tensorflow_federated as tff
import datetime


emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

BATCH_SIZE = 20
EPOCHS_PER_ROUND = 1
SHUFFLE_BUFFER = 500
NUM_CLIENTS = 10
NUM_ROUNDS = 10000
CLIENTS_PER_ROUND = 5
DAMPING_FACTOR = 0.5
LAMBDA = 0.01
PRIOR_SCALE = 100
LEARNING_RATE = 0.001
FEDAVG_INIT = False


def preprocess(dataset):
    def element_fn(element):
        return (tf.reshape(element['pixels'], [-1]),
                (tf.reshape(element['label'], [1])))

    return dataset.map(element_fn).shuffle(
        SHUFFLE_BUFFER).batch(BATCH_SIZE)


def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]


sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients)
federated_test_data = make_federated_data(emnist_test, sample_clients)

layer = DenseReparametrizationShared


def create_model(model_class, train_size):
    return model_class([
                        layer(200, input_shape=(784,), activation='relu',
                              kernel_divergence_fn=(lambda q, p, ignore: kl_lib.kl_divergence(q, p)/float(train_size)),
                              num_clients=NUM_CLIENTS, prior_scale=PRIOR_SCALE),
                        layer(200, activation='relu',
                              kernel_divergence_fn=(lambda q, p, ignore: kl_lib.kl_divergence(q, p) / float(train_size)),
                              num_clients=NUM_CLIENTS, prior_scale=PRIOR_SCALE),
                        layer(10, activation='softmax',
                              kernel_divergence_fn=(lambda q, p, ignore: kl_lib.kl_divergence(q, p) / float(train_size)),
                              num_clients=NUM_CLIENTS, prior_scale=PRIOR_SCALE)
                        ])


def compile_model(model):

    def loss_fn(y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) + sum(model.losses)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=loss_fn,
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    return model


def model_fn(model_class):
    return compile_model(create_model(model_class))


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/virtual_' + current_time
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

virtual_process = VirtualFedProcess(model_fn, NUM_CLIENTS, fed_avg_init=FEDAVG_INIT)
virtual_process.fit(federated_train_data, NUM_ROUNDS, CLIENTS_PER_ROUND, EPOCHS_PER_ROUND,
                    federated_test_data=federated_test_data)
