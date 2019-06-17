# VIRTUAL

The VIRTUAL package implements a model to perform federated multitask learning with with variational NN. The model is described in details at https://arxiv.org/abs/1906.06268


## Getting Started

We recommend to setup Miniconda to create a python environment.

```
conda env create -f environment virtual
source activate virtual
```

## Structure

### [Server](source/server.py)

The file server contains the class Server, subclass of a keras sequential model tf.kerar.Sequential.
Initialize a Server model like any other sequential model. The layer that have "lateral" in the name are going to be used as lateral connection to all the clients:

```python
from server import Server
import tensorflow as tf
import tensorflow_probability as tfp

layer = tfp.layers.DenseReparameterization
SIZE_DATASET = 60000
with tf.variable_scope('server'):
    server = Server([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Flatten(),
            layer(100, activation='relu', name='lateral',
                  kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p)/SIZE_DATASET),
            layer(10, activation='softmax',
                  kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p)/SIZE_DATASET)
	])

```

### [NetworkManager](source/network_manager.py)

The network_manager file contains the class NetworkManager, that manages the bayesian network of Server and Clients, from its creation to the message massing. To the user it resembles a keras model.
A usage example: we first create the network manager feedingn the server that we created before:

```python
network_m = NetworkManager(server)
```

Then we create as many clients as we need:

```python
CLIENTS = 5
clients = network_m.create_clients(CLIENTS)
```

Now we can compile the network you would do for a keras model
```python
network_m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])
```
The optimizer, loss and metrics are going to be applied on every client output. 
To fit a network we have now to specify a list of datasets in the usual keras model fashion. You also have to define the sequence of models and datasets to fit. 
An example reads: 

```python
x = []
x_t = []
y = []
y_t = []

for _ in range(CLIENTS):
    x_train_perm, x_test_perm = permuted_mnist(x_train, x_test)
    x.append(x_train_perm)
    x_t.append(x_test_perm)
    y.append(y_train)
    y_t.append(y_test)

model_sequence = [0, 1, 0, 2, 0, 2, 2]
data_sequence = [0, 1, 0, 2, 0, 2, 3]

network_m.fit(model_sequence, data_sequence,
            x=x, y=y, epochs=1, validation_data=zip(x_t, y_t), batch_size=128)
'''