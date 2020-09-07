import os
import logging

import tensorflow as tf
import numpy as np
import GPUtil

from source.constants import ROOT_LOGGER_STR


logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


def gpu_session(num_gpus=None, gpus=None):
    print(gpus, tf.config.experimental.list_physical_devices('GPU'))
    if gpus:
        logger.info(f"{gpus}, "
                    f"{tf.config.experimental.list_physical_devices('GPU')}")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    elif num_gpus:
        if num_gpus > 0:
            os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
            os.environ["CUDA_VISIBLE_DEVICES"] = set_free_gpus(num_gpus)
            print('visible devices:', os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"])
    gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    logger.info(f'Cuda devices: {gpus}') if gpus else \
        logger.info('No Cuda devices')

    if gpus or num_gpus > 0:
        logger.info(f"{gpus}, "
                    f"{tf.config.experimental.list_physical_devices('GPU')}")
        gpus = [tf.config.experimental.list_physical_devices('GPU')[int(gpu)] for gpu in gpus]
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        tf.config.set_soft_device_placement(True)
        [tf.config.experimental.set_memory_growth(gpu, enable=True) for gpu in gpus]


def set_free_gpus(num):
    # num: integer; number of GPUs that shall be allocated
    # returns: string; listing a total of 'num' available GPUs.

    list_gpu = GPUtil.getAvailable(limit=num, maxMemory=0.01)
    return str(list_gpu)[1:-1]


def avg_dict(history_list, cards):
    avg_dict = {}
    for el in history_list:
        if hasattr(el, 'keys'):
            keys = el.keys()
            continue
    for key in keys:
        lists = list(zip(*[(history[key][-1]*card, card)
                           for history, card in zip(history_list, cards)
                           if history]))
        avg_dict[key] = sum(lists[0])/sum(lists[1])
    return avg_dict


def avg_dict_eval(eval_fed, cards):
    eval = np.array([np.array(eval)*card for eval, card in zip(eval_fed, cards)])
    return eval.sum(axis=0)


class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

    def __init__(self, name='sparse_categorical_accuracy', dtype=None, vocab_size=0):
        super().__init__(name, dtype=dtype)
        self.vocab_size = vocab_size

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1, 1])
        y_pred = tf.reshape(y_pred, [-1, self.vocab_size+1, 1])
        if sample_weight is not None:
            sample_weight = tf.reshape(sample_weight, [-1, 1])
        return super().update_state(
            y_true, y_pred, sample_weight)