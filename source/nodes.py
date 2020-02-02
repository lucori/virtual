import tensorflow as tf
from dense_reparametrization_shared import DenseReparametrizationShared


class Client(tf.keras.Sequential):

    def __init__(self, layers=None, name=None):
        super(Client, self).__init__(layers=layers, name=name)
        self.s_i_to_update = False

    def compute_delta(self):
        '''return a list of delta, one delta (loc and scale tensors) per layer'''

        delta = []
        for layer in self.layers:
            if isinstance(layer, DenseReparametrizationShared):
                delta.append(layer.compute_delta())
        return delta

    def renew_s_i(self):
        if self.s_i_to_update:
            for layer in self.layers:
                if isinstance(layer, DenseReparametrizationShared):
                    layer.renew_s_i()
        else:
            self.s_i_to_update = True

    def receive_s(self, server):
        for lc, ls in zip(self.layers, server.layers):
            if isinstance(lc, DenseReparametrizationShared) and isinstance(ls, DenseReparametrizationShared):
                lc.receive_s(ls)

    def apply_damping(self, damping_factor):
        for layer in self.layers:
            if isinstance(layer, DenseReparametrizationShared):
                layer.apply_damping(damping_factor)

    def initialize_kernel_posterior(self):
        for layer in self.layers:
            if isinstance(layer, DenseReparametrizationShared):
                layer.initialize_kernel_posterior()


class Server(tf.keras.Sequential):

    def __init__(self, layers=None, name=None):
        super(Server, self).__init__(layers=layers, name=name)

    def apply_delta(self, delta):
        for i, layer in enumerate(x for x in self.layers if isinstance(x, DenseReparametrizationShared)):
            if isinstance(layer, DenseReparametrizationShared):
                layer.apply_delta(delta[i])


