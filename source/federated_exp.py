import tensorflow as tf
from utils import gpu_session, permuted_mnist_for_n_tasks, get_mlp_server, femnist_data, DenseReparameterizationPriorUpdate
from network_manager import NetworkManager
import numpy as np
import pickle

NUM_GPUS = 1
NUM_TASKS = 5
NUM_EPOCHS = 10000
BATCH_SIZE = 32
NUM_REP = 5
DATA_SET_SIZE_PER_USER = 250
TEST_SET_SIZE_PER_USER = 25
NUM_SAMPLES = 10
NUM_REFINEMENT = 2
INPUT_SHAPE = (784, )
LAYER = DenseReparameterizationPriorUpdate
LAYER_UNITS = [100, 100, 62]
ACTIVATIONS = ['relu', 'relu', 'softmax']
LEARNING_RATE = 0.001
PATIENCE = 100
VALIDATION_SPLIT = 0.2
TRAIN_SET_SIZE_PER_USER = int(DATA_SET_SIZE_PER_USER*(1-VALIDATION_SPLIT))
DATA_SET_GENERATOR = femnist_data
VERBOSE = 0

config = gpu_session(num_gpus=1)
if config:
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)

accuracy_virtual = []
history_overall = []

x, y, x_t, y_t = DATA_SET_GENERATOR(NUM_TASKS, global_data=False,
                              train_set_size_per_user=DATA_SET_SIZE_PER_USER,
                              test_set_size_per_user=TEST_SET_SIZE_PER_USER)

network_m = NetworkManager(
    get_mlp_server(INPUT_SHAPE, LAYER, LAYER_UNITS, ACTIVATIONS, TRAIN_SET_SIZE_PER_USER, NUM_SAMPLES),
                        data_set_size=TRAIN_SET_SIZE_PER_USER, n_samples=NUM_SAMPLES)
clients = network_m.create_clients(NUM_TASKS)
network_m.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])
model_sequence = np.tile(range(NUM_TASKS), NUM_REFINEMENT)
data_sequence = np.tile(range(NUM_TASKS), NUM_REFINEMENT)
history_virtual, evaluation_virtual = network_m.fit(model_sequence, data_sequence,
                        x=x, y=y, validation_split=VALIDATION_SPLIT,
                        epochs=NUM_EPOCHS, test_data=zip(x_t, y_t), batch_size=BATCH_SIZE,
                        callbacks=[early_stopping], verbose=VERBOSE
                        )

print('virtual done')
with open('femnist_virtual_history.pkl', 'wb') as f:
    pickle.dump(history_virtual, f)

with open('femnist_virtual_eval.pkl', 'wb') as f:
    pickle.dump(evaluation_virtual, f)

history_local = {}
evaluation_local = {}
for n in range(NUM_TASKS):
    print('local model number ', n)
    single_model = get_mlp_server(INPUT_SHAPE, LAYER, LAYER_UNITS, ACTIVATIONS, TRAIN_SET_SIZE_PER_USER, NUM_SAMPLES)
    single_model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=[tf.keras.metrics.categorical_accuracy])
    history_local[n] = single_model.fit(x=x[n], y=y[n], validation_split=VALIDATION_SPLIT,
                        epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[early_stopping], verbose=VERBOSE).history
    evaluation_local[n] = single_model.evaluate(x_t[n], y_t[n])
    print(evaluation_local[n])

with open('femnist_local_history.pkl', 'wb') as f:
    pickle.dump(history_local, f)

with open('femnist_local_eval.pkl', 'wb') as f:
    pickle.dump(evaluation_local, f)

x = np.concatenate(x)
y = np.concatenate(y)
x_t = np.concatenate(x_t)
y_t = np.concatenate(y_t)

DATA_SET_SIZE = TRAIN_SET_SIZE_PER_USER*NUM_TASKS
global_net = get_mlp_server(INPUT_SHAPE, LAYER, LAYER_UNITS, ACTIVATIONS, DATA_SET_SIZE, NUM_SAMPLES)
global_net.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])
print('starting global model')
history_global = global_net.fit(x=x,
                         y=y,
                         epochs=NUM_EPOCHS,
                         validation_split=VALIDATION_SPLIT,
                         batch_size=BATCH_SIZE,
                         callbacks=[early_stopping], verbose=VERBOSE
                         ).history
evaluation_global = global_net.evaluate(x_t, y_t)
print(evaluation_global)

with open('femnist_global_history.pkl', 'wb') as f:
    pickle.dump(history_global, f)

with open('femnist_global_eval.pkl', 'wb') as f:
    pickle.dump(evaluation_global, f)

print('done')