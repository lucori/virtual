import tensorflow as tf
import numpy as np


def mnist(num_tasks=1, global_data=False, train_set_size_per_user=-1, test_set_size_per_user=-1):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = tf.keras.utils.to_categorical(y_train.astype('int32'))
    y_test = tf.keras.utils.to_categorical(y_test.astype('int32'))
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    x_train /= 126
    x_test /= 126

    if not global_data:
        x_train = np.split(x_train, num_tasks)
        y_train = np.split(y_train, num_tasks)
        x_test = [x_test for _ in range(num_tasks)]
        y_test = [y_test for _ in range(num_tasks)]
        x_train = [x[:train_set_size_per_user] for x in x_train]
        y_train = [y[:train_set_size_per_user] for y in y_train]
        x_test = [x[:test_set_size_per_user] for x in x_test]
        y_test = [y[:test_set_size_per_user] for y in y_test]

    return x_train, y_train, x_test, y_test


def permute(x_train, x_test):

    def shuffle(x, indx):
        shape = x[0].shape
        for i, _ in enumerate(x):
            x[i] = (x[i].flatten()[indx]).reshape(shape)
        return x
    indx = np.random.permutation(x_train[0].size)
    x_train = shuffle(x_train, indx)
    x_test = shuffle(x_test, indx)
    return x_train, x_test


def permuted_mnist(num_tasks, global_data=False, train_set_size_per_user=-1, test_set_size_per_user=-1):
    x_train, y_train, x_test, y_test = mnist(global_data=True)
    x = []
    x_t = []
    y = []
    y_t = []
    x_train = x_train[:train_set_size_per_user]
    y_train = y_train[:train_set_size_per_user]
    x_test = x_test[:test_set_size_per_user]
    y_test = y_test[:test_set_size_per_user]
    for _ in range(num_tasks):
        x_train_perm, x_test_perm = permute(x_train, x_test)
        x.append(x_train_perm)
        x_t.append(x_test_perm)
        y.append(y_train)
        y_t.append(y_test)

    if global_data:
        x = np.concatenate(x)
        y = np.concatenate(y)
        x_t = np.concatenate(x_t)
        y_t = np.concatenate(y_t)

    return x, y, x_t, y_t


def femnist(num_tasks, global_data=False, train_set_size_per_user=-1, test_set_size_per_user=-1):
    import os
    import json
    data = []
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
    dir_path_train = os.path.join(dir_path, 'leaf/data/femnist/data/train')
    dir_path_test = os.path.join(dir_path, 'leaf/data/femnist/data/test')

    for file_name in os.listdir(dir_path_train):
        file = os.path.join(dir_path_train, file_name)
        file = open(file)
        data.append(json.load(file))
        file.close()
    data_merge_train = {'users': [], 'num_samples': [], 'user_data': {}}
    for d in data:
        data_merge_train['users'] = data_merge_train['users'] + d['users']
        data_merge_train['num_samples']= data_merge_train['num_samples'] + d['num_samples']
        data_merge_train['user_data'].update(d['user_data'])

    data = []
    for file_name in os.listdir(dir_path_test):
        file = os.path.join(dir_path_test, file_name)
        file = open(file)
        data.append(json.load(file))
        file.close()
    data_merge_test = {'users': [], 'num_samples': [], 'user_data': {}}
    for d in data:
        data_merge_test['users'] = data_merge_test['users'] + d['users']
        data_merge_test['num_samples'] = data_merge_test['num_samples'] + d['num_samples']
        data_merge_test['user_data'].update(d['user_data'])

    x = []
    y = []
    xt = []
    yt = []

    users = [u for u, n in zip(data_merge_train['users'], data_merge_train['num_samples']) if n >= train_set_size_per_user]
    users = users[0:num_tasks]
    for user in users:
        x.append(np.array(data_merge_train['user_data'][user]['x'])[0:train_set_size_per_user])
        y.append(tf.keras.utils.to_categorical(np.array(data_merge_train['user_data'][user]['y']),
                                               num_classes=62)[0:train_set_size_per_user])
        xt.append(np.array(data_merge_test['user_data'][user]['x'])[0:test_set_size_per_user])
        yt.append(tf.keras.utils.to_categorical(np.array(data_merge_test['user_data'][user]['y']),
                                                num_classes=62)[0:test_set_size_per_user])
    if global_data:
        x = np.concatenate(x)
        y = np.concatenate(y)
        xt = np.concatenate(xt)
        yt = np.concatenate(yt)

    return x, y, xt, yt