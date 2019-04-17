import tensorflow as tf
import tensorflow_probability as tfp
from utils import gpu_session, permuted_mnist, mnist_data
from network_manager import NetworkManager
from server import Server

layer = tfp.layers.DenseReparameterization
x_train,y_train,x_test,y_test = mnist_data()

NUM_GPUS = 1

config = gpu_session(NUM_GPUS)
if config:
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

with tf.variable_scope('server'):
    server = Server([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            layer(100, activation='relu', name='lateral',
                  kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p)/60000),
            layer(10, activation='softmax',
                  kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p)/60000)
    ])

x = []
x_t = []
y = []
y_t = []

CLIENTS = 5
for _ in range(CLIENTS):
    x_train_perm, x_test_perm = permuted_mnist(x_train, x_test)
    x.append(x_train_perm)
    x_t.append(x_test_perm)
    y.append(y_train)
    y_t.append(y_test)

network_m = NetworkManager(server)
clients = network_m.create_clients(CLIENTS)

network_m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

model_sequence = [0, 1, 0, 2, 0, 2, 2]
data_sequence = [0, 1, 0, 2, 0, 2, 3]

network_m.fit(model_sequence, data_sequence,
            x=x, y=y, epochs=1, validation_data=zip(x_t, y_t), batch_size=128)
