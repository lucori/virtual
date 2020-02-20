import time
import os
from data_utils import post_process_datasets, federated_dataset
import tensorflow as tf
from experiment_utils import run_simulation, get_compiled_model_fn_from_dict
from tensorboard.plugins.hparams import api as hp
import datetime
from utils import KLWeightingScheduler, gpu_session

dir_path = os.path.dirname(os.path.realpath(__file__))
inf = int(1e10)

gpu_session(1)

# current time for file names
time = time.strftime("%Y%m%d-%H%M%S")
print("Time:", time)
data_set_conf = {'name': 'shakespeare',
                 'num_clients': 10}

training_conf = {'method': 'fedavg',
                 'batch_size': 20,
                 'epochs_per_round': 1,
                 'num_rounds': 100,
                 'clients_per_round': 5,
                 'optimizer': 'sgd',
                 'fed_avg_init': True,
                 'tensorboard_updates': 1,
                 'learning_rate': 0.001,
                 'damping_factor': 1.,
                 'kl_weighting': 1/1e5
                 }

model_conf = {'input_shape': (784,),
              'layer': 'DenseReparametrizationShared',
              'layer_units': [100, 100, 10],
              'activations': ['relu', 'relu', 'softmax'],
              'prior_scale': 1.,
              'hierarchical': False,
              }

federated_train_data, federated_test_data = federated_dataset(data_set_conf['name'],
                                                              num_clients=data_set_conf['num_clients'])
train_size = [tf.data.experimental.cardinality(data).numpy() for data in federated_train_data]
test_size = [tf.data.experimental.cardinality(data).numpy() for data in federated_test_data]

if training_conf['method'] == 'virtual':
    epochs = 1
else:
    epochs = training_conf['epochs_per_round']

federated_train_data = post_process_datasets(federated_train_data, training_conf['batch_size'], epochs)
federated_test_data = post_process_datasets(federated_test_data, training_conf['batch_size'])

num_clients = len(federated_train_data)
model_conf['num_clients'] = num_clients

sample_batch = tf.nest.map_structure(
    lambda x: x.numpy(), iter(federated_train_data[0]).next())

HP_KL_WEIGHTING = hp.HParam('kl_weighting', hp.Discrete([1/1e4]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([10]))
HP_EPOCHS_PER_ROUND = hp.HParam('epochs_per_round', hp.Discrete([1]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.002]))

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

logdir = 'logs/' + data_set_conf['name'] + '_' + training_conf['method'] + '_' + current_time

with tf.summary.create_file_writer(logdir).as_default():
    hp.hparams_config(hparams=[HP_KL_WEIGHTING, HP_BATCH_SIZE, HP_EPOCHS_PER_ROUND, HP_OPTIMIZER, HP_LEARNING_RATE],
                      metrics=[hp.Metric('sparse_categorical_accuracy', display_name='Accuracy')],
                      )

session_num = 0

for kl_weighting in HP_KL_WEIGHTING.domain.values:
    for batch_size in HP_BATCH_SIZE.domain.values:
        for epochs_per_round in HP_EPOCHS_PER_ROUND.domain.values:
            for optimizer in HP_OPTIMIZER.domain.values:
                for learning_rate in HP_LEARNING_RATE.domain.values:
                    training_conf['kl_weighting'] = kl_weighting
                    training_conf['batch_size'] = batch_size
                    training_conf['epochs_per_round'] = epochs_per_round
                    training_conf['optimizer'] = optimizer
                    training_conf['learning_rate'] = learning_rate
                    hparams = {HP_KL_WEIGHTING: kl_weighting,
                               HP_BATCH_SIZE: batch_size,
                               HP_EPOCHS_PER_ROUND: epochs_per_round,
                               HP_OPTIMIZER: optimizer,
                               HP_LEARNING_RATE: learning_rate}
                    print(hparams)
                    logdir_run = logdir + '/' + str(session_num)
                    with tf.summary.create_file_writer(logdir_run).as_default():
                        hp.hparams(hparams)
                    kl_weighting_sch = KLWeightingScheduler(kl_weighting)
                    model_conf['kl_weight'] = kl_weighting_sch.kl_weight
                    training_conf['callbacks'] = [kl_weighting_sch]
                    model_fn = get_compiled_model_fn_from_dict(model_conf, training_conf, sample_batch)
                    run_simulation(model_fn, federated_train_data, federated_test_data, train_size, test_size,
                                   model_conf, training_conf, logdir_run)
                    session_num += 1

