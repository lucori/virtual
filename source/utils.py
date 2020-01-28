import os
import GPUtil
import tensorflow as tf


def gpu_session(num_gpus=None, gpus=None):
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
        distribution = tf.distribute.MirroredStrategy()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        return config, distribution
    else:
        return None, tf.distribute.MirroredStrategy()


def set_free_gpus(num):
    # num: integer; number of GPUs that shall be allocated
    # returns: string; listing a total of 'num' available GPUs.

    list_gpu = GPUtil.getAvailable(limit=num, maxMemory=0.01)
    return str(list_gpu)[1:-1]
