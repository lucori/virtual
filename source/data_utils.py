import os
import logging
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.simulation import hdf5_client_data
from source.constants import ROOT_LOGGER_STR

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


SHUFFLE_BUFFER = 500
BUFFER_SIZE = 10000


def post_process_datasets(federated_data, epochs=1):
    return [data.repeat(epochs).shuffle(SHUFFLE_BUFFER).prefetch(BUFFER_SIZE)
            for data in federated_data]


def federated_dataset(dataset_conf, data_dir=Path('data')):
    name = dataset_conf['name']
    num_clients = dataset_conf['num_clients']
    if name == 'mnist':
        x_train, y_train, x_test, y_test = mnist_preprocess(data_dir)
        x_train = np.split(x_train, num_clients)
        y_train = np.split(y_train, num_clients)
        x_test = np.split(x_test, num_clients)
        y_test = np.split(y_test, num_clients)

        federated_train_data = post_process_datasets([tf.data.Dataset.from_tensor_slices(data)
                                                      for data in zip(x_train, y_train)])
        federated_test_data = post_process_datasets([tf.data.Dataset.from_tensor_slices(data)
                                                     for data in zip(x_test, y_test)])
        train_size = [x.shape[0] for x in x_train]
        test_size = [x.shape[0] for x in x_test]

    if name == 'femnist':
        if (data_dir
                and (data_dir / 'datasets' / 'fed_emnist_digitsonly_train.h5').is_file()
                and (data_dir / 'datasets' / 'fed_emnist_digitsonly_test.h5').is_file()):
            train_file = data_dir / 'datasets' / 'fed_emnist_digitsonly_train.h5'
            test_file = data_dir / 'datasets' / 'fed_emnist_digitsonly_test.h5'

            logger.debug(f"Data already exists, loading from {data_dir}")
            emnist_train = hdf5_client_data.HDF5ClientData(str(train_file))
            emnist_test = hdf5_client_data.HDF5ClientData(str(test_file))
        else:
            emnist_train, emnist_test = tff.simulation.datasets.emnist.\
                load_data(cache_dir=data_dir)
        post_shape = [-1]
        if 'shape' in dataset_conf:
            post_shape = dataset_conf['shape']

        def preprocess(dataset):
            def element_fn(element):
                return (tf.reshape(element['pixels'], post_shape),
                        (tf.reshape(element['label'], [1])))

            return dataset.map(element_fn)

        def make_federated_data(client_data, client_ids):
            return [preprocess(client_data.create_tf_dataset_for_client(x))
                    for x in client_ids]

        sample_clients = emnist_train.client_ids[0:num_clients]
        federated_train_data = make_federated_data(emnist_train, sample_clients)
        federated_test_data = make_federated_data(emnist_test, sample_clients)

        train_size = [tf.data.experimental.cardinality(data).numpy()
                      for data in federated_train_data]
        test_size = [tf.data.experimental.cardinality(data).numpy()
                     for data in federated_test_data]
        federated_train_data = post_process_datasets(federated_train_data)
        federated_test_data = post_process_datasets(federated_test_data)

    if name == 'shakespeare':
        federated_train_data, federated_test_data, train_size, test_size = \
            shakspeare(num_clients, dataset_conf['seq_length'], data_dir)

    if name == 'pmnist':
        federated_train_data, federated_test_data = permuted_mnist(
            num_clients=num_clients)
        train_size = [data[0].shape[0] for data in federated_train_data]
        test_size = [data[0].shape[0] for data in federated_test_data]
        federated_train_data = [tf.data.Dataset.from_tensor_slices(data)
                                for data in federated_train_data]
        federated_test_data = [tf.data.Dataset.from_tensor_slices(data)
                               for data in federated_test_data]

    if name == 'human_activity':
        x, y = human_activity_preprocess(data_dir)
        x, y, x_t, y_t = data_split(x, y)
        train_size = [xs.shape[0] for xs in x]
        test_size = [xs.shape[0] for xs in x_t]
        federated_train_data = [tf.data.Dataset.from_tensor_slices(data) for data in zip(x, y)]
        federated_test_data = [tf.data.Dataset.from_tensor_slices(data) for data in zip(x_t, y_t)]

    if name == 'vehicle_sensor':
        x, y = vehicle_sensor_preprocess(data_dir)
        x, y, x_t, y_t = data_split(x, y)
        train_size = [xs.shape[0] for xs in x]
        test_size = [xs.shape[0] for xs in x_t]
        federated_train_data = [tf.data.Dataset.from_tensor_slices(data) for data in zip(x, y)]
        federated_test_data = [tf.data.Dataset.from_tensor_slices(data) for data in zip(x_t, y_t)]

    return federated_train_data, federated_test_data, train_size, test_size


def data_split(x, y, test_size=0.25):
    x, x_t, y, y_t = zip(*[train_test_split(x_i, y_i, test_size=test_size) for x_i, y_i in zip(x, y)])
    return x, y, x_t, y_t


def mnist_preprocess(data_dir=None):
    if data_dir and (data_dir / 'datasets' / 'mnist.npz').is_file():
        file_path = data_dir / 'datasets' / 'mnist.npz'

        logger.debug(f"Data already exists, loading from {data_dir}")
        with np.load(file_path  , allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
    else:
        (x_train, y_train), (x_test, y_test) = \
            tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 784))
    x_test = x_test.reshape((-1, 784))
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test


def permute(x):

    def shuffle(a, i):
        for j, _ in enumerate(a):
            a[j] = (a[j].flatten()[i])
        return a

    if isinstance(x, list):
        indx = np.random.permutation(x[0].shape[-1])
        permuted = []
        for el in x:
            permuted.append(shuffle(el, indx))
    else:
        indx = np.random.permutation(x.shape[-1])
        permuted = shuffle(x, indx)

    return permuted


def permuted_mnist(num_clients=100):
    x_train, y_train, x_test, y_test = mnist_preprocess()
    x_train = np.split(x_train, num_clients)
    y_train = np.split(y_train, num_clients)
    x_test = np.split(x_test, num_clients)
    y_test = np.split(y_test, num_clients)

    federated_train = []
    federated_test = []
    for x, xt, y, yt in zip(x_train, x_test, y_train, y_test):
        x, xt = permute([x, xt])
        federated_train.append((x, y))
        federated_test.append((xt, yt))
    return federated_train, federated_test


def download_file(url, filename):
    import requests
    from tqdm import tqdm

    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filename, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()


# It's important that the following link does not remove the zip file.
# Otherwise the enxt time data will be downloaded again.
def human_activity_preprocess(data_dir=None):

    if not data_dir:
        data_dir = Path(__file__).parent.absolute().parent
        data_dir = data_dir / 'data' / 'human_activity'

        if not data_dir.exists():
            data_dir.mkdir(parents=True)

    subdirs = [f for f in data_dir.iterdir() if f.is_file()]
    if not subdirs:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
        zip_file = data_dir / 'original_data.zip'
        download_file(url, zip_file)

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    data_dir = data_dir / 'UCI HAR Dataset'
    data_dir_train = data_dir /  'train'
    data_dir_test = data_dir / 'test'

    x_train = pd.read_csv(data_dir_train / 'X_train.txt',
                          delim_whitespace=True, header=None).values
    y_train = pd.read_csv(data_dir_train / 'y_train.txt',
                          delim_whitespace=True, header=None).values
    task_index_train = pd.read_csv(data_dir_train / 'subject_train.txt',
                                   delim_whitespace=True, header=None).values
    x_test = pd.read_csv(data_dir_test / 'X_test.txt',
                         delim_whitespace=True, header=None).values
    y_test = pd.read_csv(data_dir_test / 'y_test.txt',
                         delim_whitespace=True, header=None).values
    task_index_test = pd.read_csv(data_dir_test / 'subject_test.txt',
                                  delim_whitespace=True, header=None).values

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test)).squeeze()
    task_index = np.concatenate((task_index_train, task_index_test)).squeeze()
    argsort = np.argsort(task_index)
    x = x[argsort]
    y = np.array(y[argsort])
    y = y-1
    task_index = task_index[argsort]
    split_index = np.where(np.roll(task_index, 1) != task_index)[0][1:]
    x = np.split(x, split_index)
    y = np.split(y, split_index)

    return x, y


# It's important that the following link does not remove the zip file.
# Otherwise the enxt time data will be downloaded again.
def vehicle_sensor_preprocess(data_dir=None):
    if not data_dir:
        data_dir = Path(__file__).parent.absolute().parent
        data_dir = data_dir / 'data' / 'vehicle_sensor'

        if not data_dir.exists():
            data_dir.mkdir(parents=True)

    subdirs = [f for f in data_dir.iterdir() if f.is_file()]
    if not subdirs:
        url = 'http://www.ecs.umass.edu/~mduarte/images/event.zip'
        zip_file = data_dir / 'original_data.zip'
        download_file(url, zip_file)

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    data_dir = data_dir / 'events' / 'runs'

    x = []
    y = []
    task_index = []
    for root, dir, file_names in os.walk(data_dir):
        if 'acoustic' not in root and 'seismic' not in root:
            x_tmp = []
            for file_name in file_names:
                if 'feat' in file_name:
                    dt_tmp = pd.read_csv(os.path.join(root, file_name),  sep=' ',
                                         skipinitialspace=True, header=None).values[:, :50]
                    x_tmp.append(dt_tmp)
            if len(x_tmp) == 2:
                x_tmp = np.concatenate(x_tmp, axis=1)
                x.append(x_tmp)
                task_index.append(int(os.path.basename(root)[1:])*np.ones(x_tmp.shape[0]))
                y.append(int('aav' in os.path.basename(os.path.dirname(root)))*np.ones(x_tmp.shape[0]))

    x = np.concatenate(x)
    y = np.concatenate(y)
    task_index = np.concatenate(task_index)
    argsort = np.argsort(task_index)
    x = x[argsort]
    y = y[argsort]
    task_index = task_index[argsort]
    split_index = np.where(np.roll(task_index, 1) != task_index)[0][1:]
    x = np.split(x, split_index)
    y = np.split(y, split_index)
    return x, y


def shakspeare(num_clients=-1, seq_lenght=80, data_dir=None):
    vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=vocab, values=tf.constant(list(range(1, len(vocab) + 1)),
                                           dtype=tf.int64)),
        default_value=tf.cast(0, tf.int64))

    def to_ids(x):
        s = tf.reshape(x['snippets'], shape=[1])
        chars = tf.strings.bytes_split(s).values
        ids = table.lookup(chars)
        return ids

    def preprocess(dataset):
        return (
            # Map ASCII chars to int64 indexes using the vocab
            dataset.map(to_ids)
                # Split into individual chars
                .unbatch())
        # Form example sequences of SEQ_LENGTH +1

    def postprocess(dataset):
        return (dataset.batch(seq_lenght + 1, drop_remainder=False)
                .shuffle(BUFFER_SIZE))

    def data(client, source):
        return postprocess(preprocess(source.create_tf_dataset_for_client(client)))

    if data_dir:
        train_file = data_dir / 'datasets' / 'shakespeare_train.h5'
        test_file = data_dir / 'datasets' / 'shakespeare_test.h5'
    if data_dir and train_file.is_file() and test_file.is_file():
        logger.debug(f"Data already exists, loading from {data_dir}")
        train_data = hdf5_client_data.HDF5ClientData(str(train_file))
        test_data = hdf5_client_data.HDF5ClientData(str(test_file))
    else:
        train_data, test_data = tff.simulation.datasets.shakespeare.load_data(
            cache_dir=data_dir)
    indx = [8, 11, 12, 17, 26, 32, 34, 43, 45, 66, 68, 72, 73,
            85, 92, 93, 98, 105, 106, 108, 110, 130, 132, 143, 150, 153,
            156, 158, 165, 169, 185, 187, 191, 199, 207, 212, 219, 227, 235,
            236, 238, 257, 264, 269, 278, 281, 283, 285, 288, 297, 301, 305,
            310, 324, 331, 340, 351, 362, 370, 373, 374, 375, 376, 383, 388,
            418, 428, 429, 432, 433, 458, 471, 474, 476, 485, 491, 492, 494,
            497, 500, 501, 507, 512, 519, 529, 543, 556, 564, 570, 573, 574,
            579, 580, 581, 593, 600, 601, 603, 604, 613, 622, 626, 627, 632,
            644, 645, 646, 648, 657, 658, 660, 663, 669, 671, 672, 676, 678,
            681, 684, 695]

    clients = [train_data.client_ids[i] for i in indx]
    clients = clients[0:num_clients]

    train_size = [len(list(preprocess(train_data.create_tf_dataset_for_client(client)))) for client in clients]
    test_size = [len(list(preprocess(test_data.create_tf_dataset_for_client(client)))) for client in clients]

    federated_train_data = [data(client, train_data) for client in clients]
    federated_test_data = [data(client, test_data) for client in clients]

    return federated_train_data, federated_test_data, train_size, test_size


def batch_dataset(dataset, batch_size, padding=None, seq_length=None):
    if not padding:
        return dataset.batch(batch_size)
    else:
        def split_input_target(chunk):
            input_text = tf.map_fn(lambda x: x[:-1], chunk)
            target_text = tf.map_fn(lambda x: x[1:], chunk)
            return (input_text, target_text)

        return dataset.padded_batch(batch_size,
                                    padded_shapes=[seq_length + 1],
                                    drop_remainder=True,
                                    padding_values=tf.cast(0, tf.int64)).map(split_input_target)
