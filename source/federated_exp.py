import tensorflow as tf
from general_utils import gpu_session, get_mlp_server, aulc, new_session
from data_utils import *
from layers import DenseReparameterizationServer, DenseFlipOutPriorUpdate
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from experiment_utils import simulation
from tensorflow.keras.optimizers import SGD, Adam

ex = Experiment("experiments")

ex.observers.append(
    FileStorageObserver.create("runs")
)

ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    session = {
        "num_gpus": 0,
        "gpus": None,
        "experiment_name": None,
        "num_runs": 0
    }
    data_set = {
        "num_tasks": 0,
        "min_data_set_size_per_user": 0,
        "max_data_set_size_per_user": None,
        "test_size": 0,
        "generator": None
    }
    training = {
        "num_epochs": 0,
        "batch_size": 0,
        "num_refining": 0,
        "patience": 0,
        "learning_rate": 0,
        "decay": 0,
        "steps_per_epoch":None,
        "validation_split": 0,
        "verbose": 0,
        "optimizer": None,
        "early_stopping": False
    }
    model = {
        "num_samples": 0,
        "input_shape": [],
        "layer": None,
        "dropout": [],
        "layer_units": [],
        "activations": []
    }
    methods = ['virtual', 'local', 'global']


@ex.capture
def get_globals(data_set, model, training):
    model['layer'] = globals()[model['layer']]
    data_set['generator'] = globals()[data_set['generator']]
    training['optimizer'] = globals()[training['optimizer']]

@ex.automain
def run(session, data_set, training, model, _run):
    get_globals()
    config, distribution = gpu_session(num_gpus=session['num_gpus'], gpus=session['gpus'])
    if training['early_stopping']:
        training['early_stopping'] = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=training['patience'],
                                                      restore_best_weights=True)
    else:
        training['early_stopping'] = None

    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    aulcs_multi_runs = []

    for i in range(1, session['num_runs']+1):
        print('run: ', i, 'out of: ', session['num_runs'])

        x, y, x_t, y_t = import_data(data_set, i-1)
        data_set['data_set_size'] = [x_i.shape[0] for x_i in x]
        print(data_set['data_set_size'], sum(data_set['data_set_size']))
        print(x[0].shape, y[0].shape)
        data_set['num_tasks'] = len(data_set['data_set_size'])
        validation_data = list(zip(x_t, y_t))
        test_data = list(zip(x_t, y_t))

        history_virtual, evaluation_virtual = simulation('virtual', x, y, test_data, config, data_set,
                                                         model, training, _run, validation_data)
        sess = new_session(sess_config=config)

        history_local, evaluation_local = simulation('local', x, y, test_data, config, data_set, model,
                                                     training, _run, validation_data)
        sess = new_session(sess_config=config)
        history_global, evaluation_global = simulation('global', x, y, test_data, config, data_set,
                                                       model, training, _run, validation_data)
        print(evaluation_global)

        sess = new_session(sess, config)

        print(evaluation_local, evaluation_virtual)
        aulcs_multi_runs.append(aulc(history_virtual))

    ex.info['aulc'] = aulcs_multi_runs

    print('done')
    print(ex.get_experiment_info())
    return
