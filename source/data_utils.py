import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


def import_data(data_set, run):
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.dirname(dir_path)
    dir_path = os.path.join(dir_path, 'data', data_set['generator'].__name__)
    os.makedirs(dir_path, exist_ok=True)
    file_name = os.path.join(dir_path, 'data' + str(run) + '.npy')
    if os.path.isfile(file_name):
        (x, y, x_t, y_t) = np.load(file_name)
    else:
        x, y, x_t, y_t = data_set['generator'](data_set['num_tasks'],
                                               min_data_set_size_per_user=data_set['min_data_set_size_per_user'],
                                               max_data_set_size_per_user=data_set['max_data_set_size_per_user'],
                                               test_size=data_set['test_size'])
        np.save(file_name, [x, y, x_t, y_t])
    return list(x), list(y), list(x_t), list(y_t)


def data_processor(x, y, num_tasks=-1, min_data_set_size_per_user=0, max_data_set_size_per_user=None,
                   test_size=0.25):
    x, y = zip(*[(x_i, y_i) for x_i, y_i in zip(x, y) if x_i.shape[0] >= min_data_set_size_per_user])
    if max_data_set_size_per_user:
        x = [x_i[:max_data_set_size_per_user] for x_i in x]
        y = [y_i[:max_data_set_size_per_user] for y_i in y]

    x = x[:num_tasks]
    y = y[:num_tasks]
    x, x_t, y, y_t = zip(*[train_test_split(x_i, y_i, test_size=test_size) for x_i, y_i in zip(x, y)])
    return x, y, x_t, y_t


def mnist(num_tasks=1, global_data=False, min_data_set_size_per_user=None, max_data_set_size_per_user=None,
          test_size=0.25):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = tf.keras.utils.to_categorical(y_train.astype('int32'))
    y_test = tf.keras.utils.to_categorical(y_test.astype('int32'))
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    x_train /= 126
    x_test /= 126
    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    if not global_data:
        x = np.split(x, num_tasks)
        y = np.split(y, num_tasks)

    if test_size:
        return data_processor(x, y, num_tasks, min_data_set_size_per_user, max_data_set_size_per_user, test_size)
    else:
        return x, y


def permute(x):

    def shuffle(a, i):
        shape = x[0].shape
        for j, _ in enumerate(a):
            a[j] = (a[j].flatten()[i]).reshape(shape)
        return a
    indx = np.random.permutation(x[0].size)
    return shuffle(x, indx)


def permuted_mnist(num_tasks=10, min_data_set_size_per_user=None, max_data_set_size_per_user=None,
                   test_size=None):
    x, y = mnist(num_tasks=num_tasks, global_data=False, test_size=None)
    x, y = zip(*[(permute(x_i), y_i) for x_i, y_i in zip(x, y)])

    return data_processor(x, y, num_tasks, min_data_set_size_per_user, max_data_set_size_per_user, test_size)


def femnist(num_tasks=-1, min_data_set_size_per_user=None, max_data_set_size_per_user=None, test_size=None):
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

    for user in data_merge_train['users']:
        x_merge = np.concatenate([np.array(data_merge_train['user_data'][user]['x']),
                                  np.array(data_merge_test['user_data'][user]['x'])])
        y_merge = np.concatenate([np.array(data_merge_train['user_data'][user]['y']),
                                  np.array(data_merge_test['user_data'][user]['y'])])
        x.append(x_merge)
        y.append(tf.keras.utils.to_categorical(y_merge, num_classes=62))

    return data_processor(x, y, num_tasks, min_data_set_size_per_user, max_data_set_size_per_user, test_size)


def human_activity(num_tasks=-1, min_data_set_size_per_user=None, max_data_set_size_per_user=None, test_size=None):
    import os
    import pandas as pd
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
    y = np.array(y[argsort])
    y = y-1
    y = tf.keras.utils.to_categorical(y, num_classes=12)
    task_index = task_index[argsort]
    split_index = np.where(np.roll(task_index, 1) != task_index)[0][1:]
    x = np.split(x, split_index)
    y = np.split(y, split_index)

    return data_processor(x, y, num_tasks, min_data_set_size_per_user, max_data_set_size_per_user, test_size)


def vehicle_sensor(num_tasks=-1, min_data_set_size_per_user=None, max_data_set_size_per_user=None, test_size=None):
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
    y = tf.keras.utils.to_categorical(y, num_classes=2)
    task_index = np.concatenate(task_index)
    argsort = np.argsort(task_index)
    x = x[argsort]
    y = y[argsort]
    task_index = task_index[argsort]
    split_index = np.where(np.roll(task_index, 1) != task_index)[0][1:]
    x = np.split(x, split_index)
    y = np.split(y, split_index)
    return data_processor(x, y, num_tasks, min_data_set_size_per_user, max_data_set_size_per_user, test_size)

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


def gleam(num_tasks=-1, min_data_set_size_per_user=None, max_data_set_size_per_user=None, test_size=None):
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
    y = [tf.keras.utils.to_categorical(y_i, num_classes=2) for y_i in y]

    return data_processor(x, y, num_tasks, min_data_set_size_per_user, max_data_set_size_per_user, test_size)