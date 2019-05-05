import tensorflow as tf
import tensorflow_probability as tfp
from utils import gpu_session, permuted_mnist_for_n_tasks
from network_manager import NetworkManager
from server import Server
import numpy as np

NUM_GPUS = 1
NUM_TASKS = 10
NUM_EPOCHS = 5000
BATCH_SIZE = 128
NUM_REP = 5
DATA_SET_SIZE = 6000
NUM_SAMPLES = 10

config = gpu_session(NUM_GPUS)
if config:
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)


layer = tfp.layers.DenseReparameterization
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

accuracy_virtual = []

for i in range(NUM_REP):
    np.random.seed(i)
    x, y, x_t, y_t = permuted_mnist_for_n_tasks(NUM_TASKS)

    network_m = NetworkManager(Server([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            layer(100, activation='relu', name='lateral',
                  kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p)/(DATA_SET_SIZE*NUM_SAMPLES)),
            layer(100, activation='relu', name='lateral_2',
                  kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p)/(DATA_SET_SIZE*NUM_SAMPLES)),
            layer(10, activation='softmax',
                  kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p)/(DATA_SET_SIZE*NUM_SAMPLES))
            ]), data_set_size=DATA_SET_SIZE, n_samples=NUM_SAMPLES)
    clients = network_m.create_clients(1)
    network_m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=[tf.keras.metrics.categorical_accuracy])
    model_sequence = [0]*NUM_TASKS
    data_sequence = range(NUM_TASKS)
    history = network_m.fit(model_sequence, data_sequence,
                            x=[img[0:DATA_SET_SIZE] for img in x], y=[lab[0:DATA_SET_SIZE] for lab in y],
                            epochs=NUM_EPOCHS, validation_data=zip(x_t, y_t), batch_size=BATCH_SIZE,
                            callbacks=[early_stopping]
                            )
    hist = [max(h.history['val_categorical_accuracy']) for h in history]
    accuracy_virtual.append(np.array(hist).squeeze())
    tf.keras.backend.clear_session()
    del network_m


accuracy_virtual = np.array(accuracy_virtual)
np.save('accuracy_virtual', accuracy_virtual)
