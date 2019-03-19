import tensorflow as tf
import tensorflow_probability as tfp


def model_loss(model):

    def bbp_loss(y_true, y_pred):
        """ Final loss calculation function to be passed to optimizer"""
        neg_log_likelihood = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        kl = sum(model.losses)
        loss = neg_log_likelihood
        return loss

    return bbp_loss


num_classes = 10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.Sequential([
  tfp.layers.DenseReparameterization(512, activation='relu'),
  tfp.layers.DenseReparameterization(512, activation='relu'),
  tfp.layers.DenseReparameterization(10, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=10)
model.evaluate(x_test, y_test)
