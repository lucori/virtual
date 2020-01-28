import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_probability as tfp
import collections
from tff_utils import aggregate_virtual, build_virtual_process
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from tfp_utils import DenseReparameterizationShared
from utils import gpu_session


emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])


NUM_CLIENTS = 10
NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500
TOT_TRAIN = 341873


def preprocess(dataset):

    def element_fn(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['pixels'], [-1])),
            ('y', tf.reshape(element['label'], [1])),
        ])

    return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
        SHUFFLE_BUFFER).batch(BATCH_SIZE)


preprocessed_example_dataset = preprocess(example_dataset)
sample_batch = tf.nest.map_structure(
    lambda x: x.numpy(), iter(preprocessed_example_dataset).next())


def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]


sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients)
layer = DenseReparameterizationShared


def create_keras_model_var():
    model = tf.keras.models.Sequential([
        layer(10, activation=tf.nn.softmax, input_shape=(784,),
        kernel_divergence_fn=(lambda q, p, ignore: kl_lib.kl_divergence(q, p) / TOT_TRAIN),
        num_clients=NUM_CLIENTS)
    ])

    return model


def compile_keras_model(model, lr):
    model.compile(
        loss=lambda x, y: tf.keras.losses.SparseCategoricalCrossentropy()(x, y) + sum(model.losses),
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


def model_fn_var():
    keras_model = create_keras_model_var()
    keras_model = compile_keras_model(keras_model, 0.2)
    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)


init_model = create_keras_model_var()
MODEL_SPEC = init_model.variables
MODEL_SPEC = collections.OrderedDict([(a.name[:-2], tf.TensorSpec(a.shape, dtype=a.dtype)) for a in MODEL_SPEC])
MODEL_TYPE = tff.to_type(MODEL_SPEC)
CLIENT_FLOAT_TYPE = tff.FederatedType(MODEL_TYPE, tff.CLIENTS)

init_model = compile_keras_model(init_model, 0.2)

#tff.framework.set_default_executor(tff.framework.create_local_executor())

iterative_process = build_virtual_process(model_fn_var)
state = iterative_process.initialize()

NUM_ROUNDS = 500
for round_num in range(NUM_ROUNDS):
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, metrics))




