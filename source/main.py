import tensorflow as tf
import tensorflow_probability as tfp
import os
import GPUtil

def model_loss(model):

    def bbp_loss(y_true, y_pred):
        neg_log_likelihood = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        kl = sum(model.losses)
        loss = neg_log_likelihood + kl
        return loss

    return bbp_loss


def set_free_gpus(num):
    # num: integer; number of GPUs that shall be allocated
    # returns: string; listing a total of 'num' available GPUs.

    list_gpu = GPUtil.getAvailable(limit=num, maxMemory=0.02)
    return str(list_gpu)[1:-1]


NUM_GPUS = 1

if NUM_GPUS > 0:
    os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = set_free_gpus(NUM_GPUS)
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

num_classes = 10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 126
x_test /= 126
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.Sequential([
  tfp.layers.DenseReparameterization(400, activation='relu'),
  tfp.layers.DenseReparameterization(400, activation='relu'),
  tfp.layers.DenseReparameterization(10, activation='softmax'),
])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss=model_loss(model),
              metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=10,batch_size=128)
model.evaluate(x_test, y_test)
