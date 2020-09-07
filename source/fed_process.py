import tensorflow as tf
from source.tfp_utils import loc_prod_from_locprec
eps = 1/tf.float32.max


class FedProcess:

    def __init__(self, model_fn, num_clients):
        self.model_fn = model_fn
        self.num_clients = num_clients
        self.clients_indx = range(self.num_clients)
        self.clients = []
        self.server = None

        self.train_summary_writer = None
        self.test_summary_writer = None
        self.valid_summary_writer = None

    def build(self, *args, **kwargs):
        pass

    def aggregate_deltas_multi_layer(self, deltas, client_weight=None):
        aggregated_deltas = []
        deltas = list(map(list, zip(*deltas)))
        for delta_layer in deltas:
            aggregated_deltas.append(
                self.aggregate_deltas_single_layer(delta_layer, client_weight))
        return aggregated_deltas

    def aggregate_deltas_single_layer(self, deltas, client_weight=None):
        for i, delta_client in enumerate(deltas):
            for key, el in delta_client.items():
                if isinstance(el, tuple):
                    (loc, prec) = el
                    if client_weight:
                        prec = prec*client_weight[i]*self.num_clients
                    loc = tf.math.multiply(loc, prec)
                    delta_client[key] = (loc, prec)
                else:
                    if client_weight:
                        delta_client[key] = (el*client_weight[i], )
                    else:
                        delta_client[key] = (el/self.num_clients, )

        deltas = {key: [dic[key] for dic in deltas] for key in deltas[0]}
        for key, lst in deltas.items():
            lst = zip(*lst)
            sum_el = []
            for i, el in enumerate(lst):
                add = tf.math.add_n(el)
                sum_el.append(add)

            if len(sum_el) == 2:
                loc = loc_prod_from_locprec(*sum_el)
                deltas[key] = (loc, sum_el[1])
            else:
                deltas[key] = sum_el[0]
        return deltas

