import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


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
    min_train_set_size_per_user = 250

    for file_name in os.listdir(dir_path_train):
        file = os.path.join(dir_path_train, file_name)
        file = open(file)
        data.append(json.load(file))
        file.close()
    data_merge_train = {'users': [], 'num_samples': [], 'user_data': {}}
    for d in data:
        data_merge_train['users'] = data_merge_train['users'] + d['users']
        data_merge_train['num_samples'] = data_merge_train['num_samples'] + d['num_samples']
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

    users = [u for u, n in zip(data_merge_train['users'], data_merge_train['num_samples'])
             if n >= min_train_set_size_per_user]
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


def human_activity(num_tasks=30, global_data=False, train_set_size_per_user=-1, test_set_size_per_user=-1):
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
    dir_path_train = os.path.join(dir_path, 'leaf/data/human_activity/Train')
    dir_path_test = os.path.join(dir_path, 'leaf/data/human_activity/Test')

    x_train = pd.read_csv(os.path.join(dir_path_train, 'X_train.txt'), delimiter=' ', header=None).values
    y_train = pd.read_csv(os.path.join(dir_path_train, 'y_train.txt'), delimiter=' ', header=None).values
    task_index_train = pd.read_csv(os.path.join(dir_path_train, 'subject_id_train.txt'), delimiter=' ', header=None).values
    x_test = pd.read_csv(os.path.join(dir_path_test, 'X_test.txt'), delimiter=' ', header=None).values
    y_test = pd.read_csv(os.path.join(dir_path_test, 'y_test.txt'), delimiter=' ', header=None).values
    task_index_test = pd.read_csv(os.path.join(dir_path_test, 'subject_id_test.txt'), delimiter=' ', header=None).values

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test)).squeeze()
    task_index = np.concatenate((task_index_train, task_index_test)).squeeze()
    argsort = np.argsort(task_index)
    x = x[argsort]
    y = y[argsort]
    task_index = task_index[argsort]
    split_index = np.where(np.roll(task_index,1)!=task_index)[0][1:]
    x = np.split(x, split_index)
    y = np.split(y, split_index)
    min_num_samples = min([y_i.shape for y_i in y])[0]
    x = [x_i[:min_num_samples, :] for x_i in x]
    y = [y_i[:min_num_samples] -1 for y_i in y]
    y = [tf.keras.utils.to_categorical(y_i, num_classes=12) for y_i in y]
    x, x_t, y, y_t = zip(*[train_test_split(x_i, y_i, test_size=0.25) for x_i, y_i in zip(x, y)])
    return x, y, x_t, y_t


def vehicle_sensor(num_tasks=None, global_data=False, train_set_size_per_user=-1, test_set_size_per_user=-1):
    import os
    import pandas as pd
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
    dir_path = os.path.join(dir_path, 'leaf/data/vehicle_sensor')

    x = []
    y = []
    task_index = []

    for root, dir, file_names in os.walk(dir_path):
        if 'acoustic' not in root and 'seismic' not in root:
            x_tmp = []
            for file_name in file_names:
                if 'feat' in file_name:
                    dt_tmp = pd.read_csv(os.path.join(root, file_name),  sep=' ', skipinitialspace=True, header=None).values[:,:50]
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
    min_num_samples = min([y_i.shape for y_i in y])[0]
    x = [x_i[:min_num_samples, :] for x_i in x]
    y = [y_i[:min_num_samples] for y_i in y]
    y = [tf.keras.utils.to_categorical(y_i, num_classes=2) for y_i in y]
    x, x_t, y, y_t = zip(*[train_test_split(x_i, y_i, test_size=0.25) for x_i, y_i in zip(x, y)])
    return x, y, x_t, y_t


def generate_gleam_data():
    import os
    import pandas as pd
    from scipy.stats import skew, kurtosis
    from librosa.feature import spectral_centroid, spectral_rolloff, delta
    from librosa.onset import onset_strength

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
    dir_path = os.path.join(dir_path, 'leaf/data/gleam')

    x = []
    y = []

    for root, dir, file_names in os.walk(dir_path):
        for file_name in file_names:
            if 'sensorData' in file_name and not file_name[0] == '.':
                x.append(pd.read_csv(os.path.join(root, file_name)))
            if 'annotate' in file_name and not file_name[0] == '.':
                y.append(pd.read_csv(os.path.join(root, file_name)))

    def crest_factor(a):
        return np.linalg.norm(a, ord=np.inf) / np.linalg.norm(a, ord=2)

    def delta_mean(a):
        return delta(a).mean()

    def delta_delta_mean(a):
        return delta(a, order=2).mean()

    def extract_feature(f, a):
        return [[f(s_s) for s_s in s] for s in a]

    def extract_x_and_y(x_i, y_i):
        time = x_i['Unix Time'].values
        sensor = x_i['Sensor'].values
        value1 = x_i['Value1'].values
        value2 = x_i['Value2'].values
        value3 = x_i['Value3'].values

        window_lenght = 60 * 1000
        time = time - time[0]
        time = np.array(time)

        masks_time = [np.logical_and(time >= i * window_lenght, time <= (i + 1) * window_lenght)
                      for i in range(int(time[-1] / window_lenght))]
        masks_sensor = [sensor == s for s in set(sensor) if 'Light' not in s]

        sens_value1 = [[value1[np.logical_and(m, s)] for s in masks_sensor] for m in masks_time]
        sens_value2 = [[value2[np.logical_and(m, s)] for s in masks_sensor] for m in masks_time]
        sens_value3 = [[value3[np.logical_and(m, s)] for s in masks_sensor] for m in masks_time]

        features = [np.mean, np.var, skew, kurtosis, crest_factor, spectral_centroid, onset_strength,
                    spectral_rolloff, delta_mean, delta_delta_mean]

        feat_1 = np.transpose(np.array([extract_feature(f, sens_value1) for f in features]), [1, 0, 2])
        feat_2 = np.transpose(np.array([extract_feature(f, sens_value2) for f in features]), [1, 0, 2])
        feat_3 = np.transpose(np.array([extract_feature(f, sens_value3) for f in features]), [1, 0, 2])

        feat_1 = np.reshape(feat_1, (feat_1.shape[0], -1))
        feat_2 = np.reshape(feat_2, (feat_1.shape[0], -1))
        feat_3 = np.reshape(feat_3, (feat_1.shape[0], -1))

        features = np.concatenate((feat_1, feat_2, feat_3), axis=1)

        labels_time = y_i['unix time'].values
        labels_time = np.array(labels_time - labels_time[0])
        activity = y_i['Activity'].values=='eat'
        status = y_i['Status']

        time_stamp_activity = labels_time[status == 'start']
        activity = list(activity[status == 'start'])
        masks_label = [np.logical_and(time >= time_stamp_activity[i], time < time_stamp_activity[i + 1])
                       for i, _ in enumerate(zip(time_stamp_activity, activity)) if i + 1 < len(time_stamp_activity)]
        masks_label.append(time >= time_stamp_activity[-1])
        activity_l = [np.where(m, a, None) for m, a in zip(masks_label, activity)]
        activity_lc = np.stack(activity_l)
        act = np.array([next(i for i in item if i is not None) for item in activity_lc.T])
        _, act_u = np.unique(act, return_inverse=True)
        labels = [np.argmax(np.bincount(act_u[m])) for m in masks_time]
        return features, labels

    x, y = zip(*[extract_x_and_y(x_i, y_i) for x_i, y_i in zip(x, y)])
    np.save(os.path.join(dir_path, 'x.npy'), x)
    np.save(os.path.join(dir_path, 'y.npy'), y)
    return x, y


def gleam(num_tasks=None, global_data=False, train_set_size_per_user=-1, test_set_size_per_user=-1):
    from sklearn.model_selection import train_test_split
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
    dir_path = os.path.join(dir_path, 'leaf/data/gleam')
    if os.path.isfile(os.path.join(dir_path, 'x.npy')) and os.path.isfile(os.path.join(dir_path, 'y.npy')):
        x = np.load(os.path.join(dir_path, 'x.npy'))
        y = np.load(os.path.join(dir_path, 'y.npy'))
    else:
        x, y = generate_gleam_data()

    y = [np.array(y_i) for y_i in y]
    min_num_samples = min([y_i.shape for y_i in y])[0]
    x = [x_i[:min_num_samples, :] for x_i in x]
    y = [y_i[:min_num_samples] for y_i in y]
    y = [tf.keras.utils.to_categorical(y_i, num_classes=2) for y_i in y]
    x, x_t, y, y_t = zip(*[train_test_split(x_i, y_i, test_size=0.25) for x_i, y_i in zip(x, y)])

    return x, y, x_t, y_t