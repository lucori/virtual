from keras.legacy import interfaces
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Optimizer
import tensorflow as tf


class LR_SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, multipliers=None, **kwargs):
        super(LR_SGD, self).__init__(**kwargs, name=self.__class__.__name__)
        with K.name_scope(self.__class__.__name__):
            self.iterations = tf.Variable(0, dtype='int64', name='iterations')
            self.learning_rate = tf.Variable(lr, name='learning_rate', dtype=tf.float32)
            self.momentum = tf.Variable(momentum, name='momentum')
            self.decay = tf.Variable(decay, name='decay')

        self.initial_decay = decay
        self.nesterov = nesterov
        self.lr_multipliers = multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):

            matched_layer = [x for x in self.lr_multipliers.keys() if x in p.name]
            if matched_layer:
                new_lr = lr * self.lr_multipliers[matched_layer[0]]
            else:
                new_lr = lr

            v = self.momentum * m - new_lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - new_lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.learning_rate)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(LR_SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def weights(self, value):
        self._weights = value


#TODO: also adam with lr multipliers
