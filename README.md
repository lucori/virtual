# VIRTUAL

The VIRTUAL package implements a model to perform federated multi-task learning with with variational NNs.


## Getting Started

We recommend to setup Miniconda to create a python environment from the enviroment file environment.yml

```
conda env create -f environment.yml
source activate virtual
```

Otherwise simply install packages in an environment of your choice:

```
pip install -r requirements.txt
```

### Run on Leonhard
You can submit several jobs on the Leonhard cluster simultanously. A job for all possible combinations of the `hp` parameters of the config file will be submitted. Please make sure that parameters defined as `hp` are removed from other parts of the config file.

For Leonhard it is necessary that you download the data beforehand in a specific folder (i.e. `$DATADIR/tff_virtual/data/keras`) and pass this folder as `data_dir`.
Below is an example where you should adapt both `result_dir` and `data_dir`:

```
module load gcc/6.3.0 python_gpu/3.7.4 hdf5/1.10.1

python main.py configurations/femnist_fedprox.json --result_dir $DATADIR/tff_virtual --data_dir $DATADIR/tff_virtual/data/keras --submit_leonhard
```

In addition, few of Leonhard's job submission parameters such as requested memory or requested time can be passed as parameters. For more detail please see `main.py` or use help `python main.py -h`.

Below is an example of submiting all jobs where 24h and 8500MB memory is requested.
```
python main.py configurations/femnist_fedprox.json --result_dir $DATADIR/tff_virtual --data_dir $DATADIR/tff_virtual/data/keras --submit_leonhard -m 8500 -t 24
```

## General usage

We first import useful packages


```python
import tensorflow as tf
from dense_reparametrization_shared import DenseReparametrizationShared
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from virtual_process import VirtualFedProcess
import tensorflow_federated as tff
```

We can now load a federated daataset from the tff package

```python
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

def preprocess(dataset):
    def element_fn(element):
        return (tf.reshape(element['pixels'], [-1]),
                (tf.reshape(element['label'], [1])))

    return dataset.map(element_fn).shuffle(
        SHUFFLE_BUFFER)


def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]


sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients)
federated_test_data = make_federated_data(emnist_test, sample_clients)

train_size = [tf.data.experimental.cardinality(data).numpy() for data in federated_train_data]
test_size = [tf.data.experimental.cardinality(data).numpy() for data in federated_test_data]

federated_train_data_batched = [data.batch(BATCH_SIZE) for data in federated_train_data]
federated_test_data_batched = [data.batch(BATCH_SIZE) for data in federated_test_data]

```
The train_size and test_size are useful for aggragating the deltas of the models, and have to be passed to the virtual fitting method. 
We can now define a function that return a compiled modell.


```python

layer = DenseReparametrizationShared

def create_model(model_class, train_size):
    return model_class([layer(100, input_shape=(784,), activation='relu',
                              kernel_divergence_fn=(lambda q, p, ignore: kl_lib.kl_divergence(q, p)/float(train_size)),
                              num_clients=NUM_CLIENTS),
                        layer(10, activation='softmax',
                              kernel_divergence_fn=(lambda q, p, ignore: kl_lib.kl_divergence(q, p)/float(train_size)),
                              num_clients=NUM_CLIENTS)
                        ])


def compile_model(model):

    def loss_fn(y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) + sum(model.losses)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=loss_fn,
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    return model


def model_fn(model_class, train_size):
    return compile_model(create_model(model_class, train_size))

```

Notice here that use used a specialized layer, called DenseReparametrizationShared, that is a subclass of DenseReparametrization given in the tfp package.
This layee has been specifically written for the implementation of Virtual, as it allow to update the kernel prior and reparametrize the posterior accordingly. 
Now we can define the paremeters of the training, and use the VirtualProcess as a tf model.

```python
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500
LEARNING_RATE = 0.005
NUM_CLIENTS = 10
EPOCHS_PER_ROUND = 1
NUM_ROUNDS = 100
CLIENTS_PER_ROUND = 5

logdir = 'logs/virtual'

virtual_process = VirtualFedProcess(model_fn, NUM_CLIENTS)
virtual_process.fit(federated_train_data_batched,
                    num_rounds=NUM_ROUNDS,
                    clients_per_round=CLIENTS_PER_ROUND,
                    epochs_per_round=EPOCHS_PER_ROUND,
                    train_size=train_size,
                    test_size=test_size,
                    logdir=logdir,
                    federated_test_data=federated_test_data_batched)
```

To monitor the training use tensorboard:

```
tensorboard --logdir logs/virtual
```

## Reproducing the experiments of the paper

To reproduce the experiments of the paper use the main.py file, giving a configuration file from the configuration folder as 

```
python main.py configurations/femnist_virtual.json 
```
