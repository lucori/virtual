import tensorflow as tf
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow_probability.python.layers import DenseReparameterization
from utils import get_posterior_from_layer, clone, sparse_array, get_refined_prior, gaussian_ratio_par
from tensorflow.python.keras.utils import generic_utils
import tensorflow_probability as tfp
from server import Server


class Client(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super(Client, self).__init__(*args, **kwargs)
        self.prior_fn_client = lambda layer: tfp.layers.default_multivariate_normal_fn
        self.prior_fn_server = get_refined_prior
        self.old_server_par = {}

    def new_server_and_client(self, server, client_refining=False, data_set=None):
        if client_refining:
            self.prior_fn_client = lambda l: tfp.layers.default_multivariate_normal_fn
            if data_set not in self.old_server_par:
                self.old_server_par[data_set] = None

        layer_map = {}  # Cache for created layers.
        tensor_map = {}  # Map {reference_tensor: corresponding_tensor}
        input_tensors = self.inputs

        for x, y in zip(self.inputs, input_tensors):
            tensor_map[x] = y

        # Iterated over every node in the reference model, in depth order.
        depth_keys = list(self._nodes_by_depth.keys())
        depth_keys.sort(reverse=True)
        for depth in depth_keys:
            nodes = self._nodes_by_depth[depth]
            for node in nodes:
                # Recover the corresponding layer.
                layer = node.outbound_layer
                # Don't call InputLayer multiple times.
                if isinstance(layer, InputLayer):
                    continue
                # Get or create layer.
                if layer not in layer_map:
                    # Clone layer.
                    if not issubclass(layer.__class__, DenseReparameterization):
                        new_layer = clone(layer)
                    elif '_client_' not in layer.name:
                        old_weights = None
                        if client_refining:
                            if self.old_server_par[data_set]:
                                old_weights = self.old_server_par[data_set][layer.name]
                        new_layer = clone(layer,
                                kernel_prior_fn=self.prior_fn_server(server.get_layer(layer.name), old_weights))
                    else:
                        new_layer = clone(layer, kernel_prior_fn=self.prior_fn_client(layer))

                    layer_map[layer] = new_layer
                    layer = new_layer
                else:
                    # Reuse previously cloned layer.
                    layer = layer_map[layer]

                # Gather inputs to call the new layer.
                reference_input_tensors = node.input_tensors
                reference_output_tensors = node.output_tensors

                # If all previous input tensors are available in tensor_map,
                # then call node.inbound_layer on them.
                computed_tensors = []
                for x in reference_input_tensors:
                    if x in tensor_map:
                        computed_tensors.append(tensor_map[x])

                if len(computed_tensors) == len(reference_input_tensors):
                    # Call layer.
                    if node.arguments:
                        kwargs = node.arguments
                    else:
                        kwargs = {}
                    if len(computed_tensors) == 1:
                        computed_tensor = computed_tensors[0]
                        output_tensors = generic_utils.to_list(layer(computed_tensor,
                                                                     **kwargs))
                        computed_tensors = [computed_tensor]
                    else:
                        computed_tensors = computed_tensors
                        output_tensors = generic_utils.to_list(layer(computed_tensors,
                                                                     **kwargs))

                    for x, y in zip(reference_output_tensors, output_tensors):
                        tensor_map[x] = y

        # Check that we did compute the model outputs,
        # then instantiate a new model from inputs and outputs.
        output_tensors = []
        for x in self.outputs:
            assert x in tensor_map, 'Could not compute output ' + str(x)
            output_tensors.append(tensor_map[x])

        client = Client(input_tensors, output_tensors, name=self.name)
        server_layers = []
        for layer in client.layers:
            if '_client_' not in layer.name:
                server_layers.append(layer)

        server = Server(server_layers)
        client.set_weights(self.get_weights())
        client.prior_fn_client = get_posterior_from_layer
        client.old_server_par = self.old_server_par
        return server, client
