import tensorflow as tf
import tensorflow_federated as tff
import collections
import datetime
import random

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])

tff.framework.set_default_executor(tff.framework.create_local_executor())


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/fedavg_' + current_time + '/train'
test_log_dir = 'logs/fedavg_' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


TOT_TRAIN = 341873
BATCH_SIZE = 20
EPOCHS_PER_ROUND = 1
SHUFFLE_BUFFER = 500
NUM_CLIENTS = 10
NUM_ROUNDS = 10000
CLIENTS_PER_ROUND = 5
DAMPING_FACTOR = 0.5
LEARNING_RATE = 0.001


def preprocess(dataset):

    def element_fn(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['pixels'], [-1])),
            ('y', tf.reshape(element['label'], [1])),
        ])

    return dataset.repeat(EPOCHS_PER_ROUND).map(element_fn).shuffle(
        SHUFFLE_BUFFER).batch(BATCH_SIZE)


preprocessed_example_dataset = preprocess(example_dataset)
sample_batch = tf.nest.map_structure(
    lambda x: x.numpy(), iter(preprocessed_example_dataset).next())


def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]


sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients)
federated_test_data = make_federated_data(emnist_test, sample_clients)
layer = tf.keras.layers.Dense


def create_keras_model():
    model = tf.keras.models.Sequential([
        layer(10, activation=tf.nn.softmax, input_shape=(784,),
              )
    ])

    return model


def compile_keras_model(model, lr):
    model.compile(
        loss=lambda x, y: tf.keras.losses.SparseCategoricalCrossentropy()(x, y) + sum(model.losses),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


def model_fn():
    keras_model = create_keras_model()
    keras_model = compile_keras_model(keras_model, LEARNING_RATE)
    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)


init_model = create_keras_model()
MODEL_SPEC = init_model.variables
MODEL_SPEC = collections.OrderedDict([(a.name[:-2], tf.TensorSpec(a.shape, dtype=a.dtype)) for a in MODEL_SPEC])
MODEL_TYPE = tff.to_type(MODEL_SPEC)
CLIENT_FLOAT_TYPE = tff.FederatedType(MODEL_TYPE, tff.CLIENTS)

init_model = compile_keras_model(init_model, 0.2)

iterative_process = tff.learning.build_federated_averaging_process(model_fn)
evaluation = tff.learning.build_federated_evaluation(model_fn)

state = iterative_process.initialize()

for round_num in range(NUM_ROUNDS):
    state, metrics = iterative_process.next(state, [federated_train_data[indx] for indx in
                                                    random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)])
    test_metrics = evaluation(state.model, federated_test_data)
    print('round {:2d}, metrics_train={}, metrics_test={}'.format(round_num, metrics, test_metrics))
    with train_summary_writer.as_default():
        for name, value in metrics._asdict().items():
            tf.summary.scalar(name, value, step=round_num)
    with test_summary_writer.as_default():
        for name, value in test_metrics._asdict().items():
            tf.summary.scalar(name, value, step=round_num)






