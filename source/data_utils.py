import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow_federated as tff


SHUFFLE_BUFFER = 500
BUFFER_SIZE = 500
SEQ_LENGTH = 20
BATCH_SIZE = 1


def post_process_datasets(federated_data, epochs=1):
    return [data.repeat(epochs).shuffle(SHUFFLE_BUFFER).prefetch(BUFFER_SIZE)
            for data in federated_data]


def federated_dataset(name=None, num_clients=100):
    if name == 'mnist':
        x_train, y_train, x_test, y_test = mnist_preprocess()
        x_train = np.split(x_train, num_clients)
        y_train = np.split(y_train, num_clients)
        x_test = np.split(x_test, num_clients)
        y_test = np.split(y_test, num_clients)

        federated_train_data = [tf.data.Dataset.from_tensor_slices(data) for data in zip(x_train, y_train)]
        federated_test_data = [tf.data.Dataset.from_tensor_slices(data) for data in zip(x_test, y_test)]

    if name == 'femnist':
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

        def preprocess(dataset):
            def element_fn(element):
                return (tf.reshape(element['pixels'], [-1]),
                        (tf.reshape(element['label'], [1])))

            return dataset.map(element_fn)

        def make_federated_data(client_data, client_ids):
            return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

        sample_clients = emnist_train.client_ids[0:num_clients]
        federated_train_data = make_federated_data(emnist_train, sample_clients)
        federated_test_data = make_federated_data(emnist_test, sample_clients)

    if name == 'shakespeare':
        vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=vocab, values=tf.constant(list(range(len(vocab))),
                                               dtype=tf.int64)),
            default_value=0)

        def to_ids(x):
            s = tf.reshape(x['snippets'], shape=[1])
            chars = tf.strings.bytes_split(s).values
            ids = table.lookup(chars)
            return ids

        def split_input_target(chunk):
            input_text = tf.map_fn(lambda x: x[:-1], chunk)
            target_text = tf.map_fn(lambda x: x[1:], chunk)
            return (input_text, target_text)

        def preprocess(dataset, sequence_length, batch_size):
            return (
                # Map ASCII chars to int64 indexes using the vocab
                dataset.map(to_ids)
                    # Split into individual chars
                    .unbatch()
                    # Form example sequences of SEQ_LENGTH +1
                    .batch(sequence_length + 1, drop_remainder=True)
                    # Shuffle and form minibatches
                    .shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)
                    # And finally split into (input, target) tuples,
                    # each of length SEQ_LENGTH.
                    .map(split_input_target))

        train_data, test_data = tff.simulation.datasets.shakespeare.load_data()

        def data(client, source, sequence_length, batch_size):
            return preprocess(source.create_tf_dataset_for_client(client), sequence_length, batch_size)

        clients = train_data.client_ids[0:num_clients]
        federated_train_data = [data(client, train_data, SEQ_LENGTH, BATCH_SIZE) for client in clients]
        federated_test_data = [data(client, test_data, SEQ_LENGTH, BATCH_SIZE) for client in clients]

    if name == 'pmnist':
        federated_train_data, federated_test_data = permuted_mnist(num_clients=num_clients)
        federated_train_data = [tf.data.Dataset.from_tensor_slices(data) for data in federated_train_data]
        federated_test_data = [tf.data.Dataset.from_tensor_slices(data) for data in federated_test_data]
    if name == 'human_activity':
        x, y = human_activity_preprocess()
        x, y,x_t, y_t = data_split(x, y)
        federated_train_data = [tf.data.Dataset.from_tensor_slices(data) for data in zip(x, y)]
        federated_test_data = [tf.data.Dataset.from_tensor_slices(data) for data in zip(x_t, y_t)]
    if name == 'vehicle_sensor':
        x, y = vehicle_sensor_preprocess()
        x, y, x_t, y_t = data_split(x, y)
        federated_train_data = [tf.data.Dataset.from_tensor_slices(data) for data in zip(x, y)]
        federated_test_data = [tf.data.Dataset.from_tensor_slices(data) for data in zip(x_t, y_t)]

    return federated_train_data, federated_test_data


def data_split(x, y, test_size=0.25):
    x, x_t, y, y_t = zip(*[train_test_split(x_i, y_i, test_size=test_size) for x_i, y_i in zip(x, y)])
    return x, y, x_t, y_t


def mnist_preprocess():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
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


def human_activity_preprocess():
    import os
    import pandas as pd
    import zipfile

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
    dir_path = os.path.join(dir_path, 'data', 'human_activity')

    if not os.listdir(dir_path):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
        zip_file = os.path.join(dir_path, 'original_data.zip')
        download_file(url, zip_file)

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(dir_path)

        os.remove(zip_file)

    dir_path = os.path.join(dir_path, 'UCI HAR Dataset')
    dir_path_train = os.path.join(dir_path, 'train')
    dir_path_test = os.path.join(dir_path, 'test')

    x_train = pd.read_csv(os.path.join(dir_path_train, 'X_train.txt'),
                          delim_whitespace=True, header=None).values
    y_train = pd.read_csv(os.path.join(dir_path_train, 'y_train.txt'),
                          delim_whitespace=True, header=None).values
    task_index_train = pd.read_csv(os.path.join(dir_path_train, 'subject_train.txt'),
                                   delim_whitespace=True, header=None).values
    x_test = pd.read_csv(os.path.join(dir_path_test, 'X_test.txt'),
                         delim_whitespace=True, header=None).values
    y_test = pd.read_csv(os.path.join(dir_path_test, 'y_test.txt'),
                         delim_whitespace=True, header=None).values
    task_index_test = pd.read_csv(os.path.join(dir_path_test, 'subject_test.txt'),
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


def vehicle_sensor_preprocess():
    import os
    import pandas as pd
    import zipfile

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
    dir_path = os.path.join(dir_path, 'data/vehicle_sensor')

    if not os.listdir(dir_path):
        url = 'http://www.ecs.umass.edu/~mduarte/images/event.zip'
        zip_file = os.path.join(dir_path, 'original_data.zip')
        download_file(url, zip_file)

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(dir_path)

        os.remove(zip_file)
    dir_path = os.path.join(dir_path, 'events', 'runs')

    x = []
    y = []
    task_index = []

    for root, dir, file_names in os.walk(dir_path):
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
