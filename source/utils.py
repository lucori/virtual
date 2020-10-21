import os
import logging
import numpy as np
import GPUtil
from source.constants import ROOT_LOGGER_STR
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.ops import summary_ops_v2

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

    tf_gpus = tf.config.experimental.list_physical_devices('GPU')
    if (gpus or num_gpus > 0) and len(tf_gpus) > 0:
        logger.info(f"{gpus}, "
                    f"{tf.config.experimental.list_physical_devices('GPU')}")
        gpus = [tf_gpus[int(gpu)] for gpu in gpus]
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        tf.config.set_soft_device_placement(True)
        [tf.config.experimental.set_memory_growth(gpu, enable=True) for gpu in gpus]


def set_free_gpus(num):
    # num: integer; number of GPUs that shall be allocated
    # returns: string; listing a total of 'num' available GPUs.

    list_gpu = GPUtil.getAvailable(limit=num, maxMemory=0.01)
    print(list_gpu)
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


class CustomTensorboard(tf.keras.callbacks.TensorBoard):

    def __init__(self, *args, **kwargs):
        super(CustomTensorboard, self).__init__(*args, **kwargs)
        self.epoch = 0

    def _log_distr(self, epoch):
        """Logs the weights of the gaussian distributions to TensorBoard."""
        writer = self._get_writer(self._train_run_name)
        with context.eager_mode(), \
             writer.as_default(), \
             summary_ops_v2.always_record_summaries():
            for layer in self.model.layers:
                layer_to_check = layer
                if hasattr(layer, 'cell'):
                    layer_to_check = layer.cell
                for weight in layer_to_check.trainable_weights:
                    if 'natural' in weight.name + layer.name:
                        tf.summary.histogram(layer.name + '/' + weight.name + '_gamma',
                                             weight[..., 0], step=epoch)
                        tf.summary.histogram(layer.name + '/' + weight.name + '_prec',
                                             weight[..., 1], step=epoch)
                    else:
                        tf.summary.histogram(layer.name + '/' + weight.name, weight, step=epoch)
                if hasattr(layer_to_check, 'recurrent_kernel_posterior'):
                    tf.summary.histogram(
                        layer.name + '/recurrent_kernel_posterior' + '_gamma_reparametrized',
                        layer_to_check.recurrent_kernel_posterior.distribution.gamma,
                        step=epoch)
                    tf.summary.histogram(
                        layer.name + '/recurrent_kernel_posterior' + '_prec_reparametrized',
                        layer_to_check.recurrent_kernel_posterior.distribution.prec,
                        step=epoch)
                if hasattr(layer_to_check, 'kernel_posterior'):
                    tf.summary.histogram(
                        layer.name + '/kernel_posterior' + '_gamma_reparametrized',
                        layer_to_check.kernel_posterior.distribution.gamma,
                        step=epoch)
                    tf.summary.histogram(
                        layer.name + '/kernel_posterior' + '_prec_reparametrized',
                        layer_to_check.kernel_posterior.distribution.prec,
                        step=epoch)
            writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = self.epoch + 1
        epoch = self.epoch
        """Runs metrics and histogram summaries at epoch end."""
        self._log_metrics(logs, prefix='', step=epoch)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_weights(epoch)
            self._log_distr(epoch)

        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self._log_embeddings(epoch)


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex