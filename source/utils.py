import os
import GPUtil
import tensorflow as tf
import numpy as np

def gpu_session(num_gpus=None, gpus=None):
    print(tf.config.experimental.list_physical_devices('GPU'))
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    elif num_gpus:
        if num_gpus > 0:
            os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
            os.environ["CUDA_VISIBLE_DEVICES"] = set_free_gpus(num_gpus)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"])
    gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    if gpus or num_gpus > 0:
        print(gpus, tf.config.experimental.list_physical_devices('GPU'))
        gpus = [tf.config.experimental.list_physical_devices('GPU')[int(gpu)] for gpu in gpus]
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        tf.config.set_soft_device_placement(True)


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
        lists = list(zip(*[(history[key][-1]*card, card) for history, card in zip(history_list, cards) if history]))
        avg_dict[key] = sum(lists[0])/sum(lists[1])
    return avg_dict


def avg_dict_eval(eval_fed, cards):
    avg_dict = {}
    eval = np.array([np.array(eval)*card for eval, card in zip(eval_fed, cards)])
    return eval.sum(axis=0)


class KLWeightingScheduler(tf.keras.callbacks.Callback):

    def __init__(self, initial_value):
        super(KLWeightingScheduler, self).__init__()
        self.kl_weight = tf.Variable(initial_value, trainable=False)

    def on_epoch_begin(self, epoch, logs=None):
        #self.kl_weight.assign((epoch/self.num_epochs)**100)
        pass