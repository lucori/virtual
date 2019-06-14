import tensorflow as tf


class _Client(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(_Client, self).__init__(*args, **kwargs)
        self.n_samples = None


class Client(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(Client, self).__init__()
        n_samples = kwargs.pop('n_samples', None)
        model = kwargs.pop('model', None)
        if n_samples:
            self.n_samples = n_samples
        else:
            self.n_samples = 10
        if model:
            self.model = model
        else:
            self.model = _Client(*args, **kwargs)
        self.model.n_samples = self.n_samples

    def call(self, inputs):
        output = []
        for _ in range(self.n_samples):
            output.append(self.model.call(inputs))

        output = tf.keras.layers.Lambda(lambda q: tf.stack(q))(output)
        output = tf.keras.layers.Lambda(lambda q: tf.reduce_sum(q, axis=0))(output)
        return output

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.model.summary(line_length, positions, print_fn)


def client_functional_api_test(model, n_samples):
    output = [tf.keras.layers.Lambda(lambda q: model.call(q))(model.input) for _ in range(n_samples)]
    output = tf.keras.layers.Lambda(lambda q: tf.stack(q))(output)
    output = tf.keras.layers.Lambda(lambda q: tf.reduce_sum(q, axis=0))(output)
    return tf.keras.Model(model.input, output)
