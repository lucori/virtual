import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.core.utils.federated_aggregations import _validate_value_on_clients, \
    _initial_values, _federated_reduce_with_func, _validate_dtype_is_numeric
from tensorflow_federated.python.common_libs import anonymous_tuple
import functools
from tensorflow_federated.python.common_libs.anonymous_tuple import map_structure, AnonymousTuple, flatten, \
    is_same_structure, pack_sequence_as, to_elements
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.tensorflow_libs import tensor_utils
from tensorflow_federated.python.learning.framework.optimizer_utils import ServerState, build_stateless_mean, \
    build_stateless_broadcaster, server_update_model
from tfp_utils import compute_gaussian_ratio, BayesianSGD
import tensorflow_probability as tfp
import collections


def map_structure_cross_function(fn, *structure):
    py_typecheck.check_callable(fn)
    if not structure:
        raise ValueError('Must provide at least one structure')

    py_typecheck.check_type(structure[0], AnonymousTuple)
    for i, other in enumerate(structure[1:]):
        if not is_same_structure(structure[0], other):
            raise TypeError('Structure at position {} is not the same '
                            'structure'.format(i))

    flat_structure = []
    flat_structure.extend([flatten(s) for s in structure])
    entries = list(zip(*flat_structure))
    s = []
    for i, v in enumerate(entries):
        (val,) = v
        if 'kernel_posterior_loc' in val.name:
            s.append(fn(*v, *entries[i + 1]))
        else:
            s.append(*v)
    return pack_sequence_as(structure[0], s)


def map_structure_virtual(func, *structure, **kwargs):
    if not callable(func):
        raise TypeError("func must be callable, got: %s" % func)

    if not structure:
        raise ValueError("Must provide at least one structure")

    check_types = kwargs.pop("check_types", True)
    expand_composites = kwargs.pop("expand_composites", False)

    if kwargs:
        raise ValueError(
            "Only valid keyword arguments are `check_types` and "
            "`expand_composites`, not: `%s`" % ("`, `".join(kwargs.keys())))

    for other in structure[1:-1]:
        tf.nest.assert_same_structure(structure[0], other, check_types=check_types,
                                      expand_composites=expand_composites)

    flat_structure = [tf.nest.flatten(s, expand_composites) for s in structure[:-1]]

    entries = list(zip(*flat_structure))
    ratio = []
    for i, x in enumerate(entries):
        if 'kernel_posterior_loc' in x[0].name:
            ratio.extend(func(x[0],
                              structure[-1][x[0].name.rsplit('/')[0] + '/s_i_loc'],
                              entries[i+1][0],
                              structure[-1][x[0].name.rsplit('/')[0] + '/s_i_untrasformed_scale']))
        elif 'kernel_posterior_untransformed_scale' in x[0].name:
            pass
        else:
            ratio.append(func(*x))

    return tf.nest.pack_sequence_as(
        structure[0], ratio,
        expand_composites=expand_composites)


def federated_reduce_with_multiple_func(value, tf_func_pointwise, tf_func_accumulate_merge, tf_func_report,
                                        zeros):
    member_type = value.type_signature.member

    @tff.tf_computation(value.type_signature.member, value.type_signature.member)
    def accumulate(current, value):
        if isinstance(member_type, tff.NamedTupleType):
            map = anonymous_tuple.map_structure
        else:
            map = tf.nest.map_structure
        value = map(tf_func_pointwise, value)
        value = map_structure_cross_function(tf.math.multiply, value)
        return map(tf_func_accumulate_merge, current, value)

    @tff.tf_computation(value.type_signature.member, value.type_signature.member)
    def merge(a, b):
        if isinstance(member_type, tff.NamedTupleType):
            map = anonymous_tuple.map_structure
        else:
            map = tf.nest.map_structure
        return map(tf_func_accumulate_merge, a, b)

    @tff.tf_computation(value.type_signature.member)
    def report(value):
        if isinstance(member_type, tff.NamedTupleType):
            map = anonymous_tuple.map_structure
        else:
            map = tf.nest.map_structure
        value = map(scale_reciprocal, value)
        value = map_structure_cross_function(tf.math.multiply, value)
        return map(tf_func_report, value)

    return tff.federated_aggregate(value, zeros, accumulate, merge, report)


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def untrasformed_scale_to_inverse_variance(tensor):
    if 'untransformed_scale' in tensor.name:
        return compose(tf.math.reciprocal, tf.math.square, tfp.bijectors.Softplus().forward)(tensor)
    else:
        return tensor


def precision_to_untransformed_scale(tensor):
    if 'Reciprocal' in tensor.name:
        return compose(tfp.bijectors.Softplus().inverse, tf.math.sqrt)(tensor)
    else:
        return tensor


def scale_reciprocal(tensor):
    if 'untransformed_scale' in tensor.name:
        return tf.math.reciprocal(tensor)
    else:
        return tensor


def zeros_fn_add(type):
    return tf.constant(0., dtype=type.dtype)


def virtual_delta(l1, l2, us1=None, us2=None):
    if us1 is None and us2 is None:
        return tf.subtract(l1, l2)
    else:
        s1 = tfp.bijectors.Softplus().forward(us1)
        s2 = tfp.bijectors.Softplus().forward(us2)
        loc_ratio, scale_ratio = compute_gaussian_ratio(l1, s1, l2, s2)
        return loc_ratio, tfp.bijectors.Softplus().inverse(scale_ratio)


def federated_virtual(value):
    _validate_value_on_clients(value)
    member_type = value.type_signature.member
    zeros = _initial_values(zeros_fn_add, member_type)
    reduce = federated_reduce_with_multiple_func(value,
                                                 untrasformed_scale_to_inverse_variance,
                                                 tf.math.add,
                                                 precision_to_untransformed_scale,
                                                 zeros)
    return reduce


def aggregate_virtual():
    return tff.utils.StatefulAggregateFn(initialize_fn=lambda: (),  # The state is an empty tuple.
                                         next_fn=lambda state, value, weight=None: (state, federated_virtual(value)))


def build_virtual_process(model_fn,
                          server_optimizer_fn=lambda: BayesianSGD(learning_rate=1.),
                          client_weight_fn=None,
                          stateful_delta_aggregate_fn=aggregate_virtual(),
                          stateful_model_broadcast_fn=None):

    def client_fed_virtual(model_fn):
        return ClientVirtual(model_fn(), client_weight_fn)

    if stateful_delta_aggregate_fn is None:
        stateful_delta_aggregate_fn = aggregate_virtual()
    else:
        py_typecheck.check_type(stateful_delta_aggregate_fn,
                                tff.utils.StatefulAggregateFn)

    if stateful_model_broadcast_fn is None:
        stateful_model_broadcast_fn = optimizer_utils.build_stateless_broadcaster()
    else:
        py_typecheck.check_type(stateful_model_broadcast_fn,
                                tff.utils.StatefulBroadcastFn)

    return build_model_delta_optimizer_process_virtual(
        model_fn, client_fed_virtual, server_optimizer_fn,
        stateful_delta_aggregate_fn, stateful_model_broadcast_fn)


def assign_to_client_virtual(a, b):
    if 's_loc' in a.name or 's_untrasformed_scale' in a.name:
        a.assign(b)


class ClientVirtual(optimizer_utils.ClientDeltaFn):
    """Client TensorFlow logic for VIRTUAL"""

    def __init__(self, model, client_weight_fn=None):
        """Creates the client computation for VIRTUAL.
        Args:
          model: A `tff.learning.TrainableModel`.
          client_weight_fn: Optional function that takes the output of
            `model.report_local_outputs` and returns a tensor that provides the
            weight in the federated average of model deltas. If not provided, the
            default is the total number of examples processed on device.
        """
        self._model = model_utils.enhance(model)
        py_typecheck.check_type(self._model, model_utils.EnhancedTrainableModel)

        if client_weight_fn is not None:
            py_typecheck.check_callable(client_weight_fn)
            self._client_weight_fn = client_weight_fn
        else:
            self._client_weight_fn = None

    @property
    def variables(self):
        return []

    @tf.function
    def __call__(self, dataset, initial_weights):
        if 'Dataset' not in str(type(dataset)):
            raise TypeError('Expected a data set, found {}.'.format(
                py_typecheck.type_string(type(dataset))))

        model = self._model

        #assign the new s from server to the client
        tf.nest.map_structure(assign_to_client_virtual, model.weights,
                              initial_weights)

        @tf.function
        def reduce_fn(num_examples_sum, batch):
            """Runs `tff.learning.Model.train_on_batch` on local client batch."""
            output = model.train_on_batch(batch)
            if output.num_examples is None:
                return num_examples_sum + tf.shape(output.predictions)[0]
            else:
                return num_examples_sum + output.num_examples

        num_examples_sum = dataset.reduce(
            initial_state=tf.constant(0), reduce_func=reduce_fn)

        #compute bayesian delta
        weights_delta = map_structure_virtual(virtual_delta, model.weights.trainable,
                                              initial_weights.trainable, model.weights.non_trainable)
        aggregated_outputs = model.report_local_outputs()

        #renew s_i
        for e in model.weights.trainable.keys():
            if 'kernel_posterior_loc' in e:
                model.weights.non_trainable[e.rsplit('/')[0] + '/s_i_loc'].assign(model.weights.trainable[e])
            if 'kernel_posterior_untransformed_scale' in e:
                model.weights.non_trainable[e.rsplit('/')[0] + '/s_i_untrasformed_scale'].assign(model.weights.trainable[e])


        weights_delta, has_non_finite_delta = (
            tensor_utils.zero_all_if_any_non_finite(weights_delta))
        if self._client_weight_fn is None:
            weights_delta_weight = tf.cast(num_examples_sum, tf.float32)
        else:
            weights_delta_weight = self._client_weight_fn(aggregated_outputs)
        # Zero out the weight if there are any non-finite values.
        if has_non_finite_delta > 0:
            weights_delta_weight = tf.constant(0.0)

        return optimizer_utils.ClientOutput(
            weights_delta, weights_delta_weight, aggregated_outputs,
            tensor_utils.to_odict({
                'num_examples': num_examples_sum,
                'has_non_finite_delta': has_non_finite_delta,
            }))


def server_init_virtual(model_fn, optimizer_fn, delta_aggregate_state,
                model_broadcast_state):
    """Returns initial `tff.learning.framework.ServerState`.
    Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    optimizer_fn: A no-arg function that returns a `tf.train.Optimizer`.
    delta_aggregate_state: The initial state of the delta_aggregator.
    model_broadcast_state: The initial state of the model_broadcaster.
    Returns:
    A `tff.learning.framework.ServerState` namedtuple.
    """
    model = model_utils.enhance(model_fn())
    optimizer = optimizer_fn()
    _, optimizer_vars = _build_server_optimizer_virtual(model, optimizer)
    return ServerState(
          model=model.weights,
          optimizer_state=optimizer_vars,
          delta_aggregate_state=delta_aggregate_state,
          model_broadcast_state=model_broadcast_state)


def build_model_delta_optimizer_process_virtual(model_fn,
                                                model_to_client_delta_fn,
                                                server_optimizer_fn,
                                                stateful_delta_aggregate_fn=aggregate_virtual(),
                                                stateful_model_broadcast_fn=build_stateless_broadcaster()):

    py_typecheck.check_callable(model_fn)
    py_typecheck.check_callable(model_to_client_delta_fn)
    py_typecheck.check_callable(server_optimizer_fn)
    py_typecheck.check_type(stateful_delta_aggregate_fn,
                          tff.utils.StatefulAggregateFn)
    py_typecheck.check_type(stateful_model_broadcast_fn,
                          tff.utils.StatefulBroadcastFn)

    with tf.Graph().as_default():
        dummy_model_for_metadata = model_utils.enhance(model_fn())

    @tff.federated_computation
    def server_init_tff():
        """Orchestration logic for server model initialization."""

        @tff.tf_computation
        def _fn():
            return server_init_virtual(model_fn, server_optimizer_fn,
                               stateful_delta_aggregate_fn.initialize(),
                               stateful_model_broadcast_fn.initialize())

        return tff.federated_value(_fn(), tff.SERVER)

    federated_server_state_type = server_init_tff.type_signature.result
    server_state_type = federated_server_state_type.member

    tf_dataset_type = tff.SequenceType(dummy_model_for_metadata.input_spec)
    federated_dataset_type = tff.FederatedType(
        tf_dataset_type, tff.CLIENTS, all_equal=False)

    @tff.federated_computation(federated_server_state_type,
                               federated_dataset_type)
    def run_one_round_tff(server_state, federated_dataset):
        """Orchestration logic for one round of optimization.

        Args:
          server_state: a `tff.learning.framework.ServerState` named tuple.
          federated_dataset: a federated `tf.Dataset` with placement tff.CLIENTS.

        Returns:
          A tuple of updated `tff.learning.framework.ServerState` and the result of
        `tff.learning.Model.federated_output_computation`.
        """
        model_weights_type = federated_server_state_type.member.model

        @tff.tf_computation(tf_dataset_type, model_weights_type)
        def client_delta_tf(tf_dataset, initial_model_weights):
            """Performs client local model optimization.

            Args:
            tf_dataset: a `tf.data.Dataset` that provides training examples.
            initial_model_weights: a `model_utils.ModelWeights` containing the
              starting weights.

            Returns:
            A `ClientOutput` structure.
            """
            client_delta_fn = model_to_client_delta_fn(model_fn)

            if isinstance(initial_model_weights, anonymous_tuple.AnonymousTuple):
                initial_model_weights = model_utils.ModelWeights.from_tff_value(initial_model_weights)

            client_output = client_delta_fn(tf_dataset, initial_model_weights)
            return client_output

        new_broadcaster_state, client_model = stateful_model_broadcast_fn(
            server_state.model_broadcast_state, server_state.model)

        client_outputs = tff.federated_map(client_delta_tf,
                                           (federated_dataset, client_model))

        @tff.tf_computation(
            server_state_type, model_weights_type.trainable,
            server_state.delta_aggregate_state.type_signature.member,
            server_state.model_broadcast_state.type_signature.member)
        def server_update_tf(server_state, model_delta, new_delta_aggregate_state,
                             new_broadcaster_state):
            py_typecheck.check_type(model_delta, anonymous_tuple.AnonymousTuple)
            model_delta = anonymous_tuple.to_odict(model_delta)
            py_typecheck.check_type(server_state, anonymous_tuple.AnonymousTuple)
            server_state = ServerState(
                model=model_utils.ModelWeights.from_tff_value(server_state.model),
                optimizer_state=list(server_state.optimizer_state),
                delta_aggregate_state=new_delta_aggregate_state,
                model_broadcast_state=new_broadcaster_state)

            return server_update_model_virtual(
                server_state,
                model_delta,
                model_fn=model_fn,
                optimizer_fn=server_optimizer_fn)

        fed_weight_type = client_outputs.weights_delta_weight.type_signature.member
        py_typecheck.check_type(fed_weight_type, tff.TensorType)
        if fed_weight_type.dtype.is_integer:

            @tff.tf_computation(fed_weight_type)
            def _cast_to_float(x):
                return tf.cast(x, tf.float32)

            weight_denom = tff.federated_map(_cast_to_float,
                                             client_outputs.weights_delta_weight)
        else:
            weight_denom = client_outputs.weights_delta_weight

        new_delta_aggregate_state, round_model_delta = stateful_delta_aggregate_fn(
            server_state.delta_aggregate_state,
            client_outputs.weights_delta,
            weight=weight_denom)

        server_state = tff.federated_apply(
            server_update_tf, (server_state, round_model_delta,
                               new_delta_aggregate_state, new_broadcaster_state))

        aggregated_outputs = dummy_model_for_metadata.federated_output_computation(
            client_outputs.model_output)

        aggregated_outputs = tff.federated_zip(aggregated_outputs)

        return server_state, aggregated_outputs

    return tff.utils.IterativeProcess(
      initialize_fn=server_init_tff, next_fn=run_one_round_tff)


def server_update_model_virtual(server_state, weights_delta, model_fn, optimizer_fn):
    py_typecheck.check_type(server_state, ServerState)
    py_typecheck.check_type(weights_delta, collections.OrderedDict)
    model = model_utils.enhance(model_fn())
    optimizer = optimizer_fn()
    apply_delta_fn, optimizer_vars = _build_server_optimizer_virtual(model, optimizer)

    # We might have a NaN value e.g. if all of the clients processed
    # had no data, so the denominator in the federated_mean is zero.
    # If we see any NaNs, zero out the whole update.
    no_nan_weights_delta, _ = tensor_utils.zero_all_if_any_non_finite(
                                             weights_delta)

    @tf.function
    def update_model_inner():
        """Applies the update."""
        tf.nest.map_structure(lambda a, b: a.assign(b),
                              (model.weights, optimizer_vars),
                              (server_state.model, server_state.optimizer_state))
        apply_delta_fn(no_nan_weights_delta)
        return model.weights, optimizer_vars

    model_weights, optimizer_vars = update_model_inner()

    return tff.utils.update_state(
        server_state, model=model_weights, optimizer_state=optimizer_vars)


def _build_server_optimizer_virtual(model, optimizer):
    @tf.function
    def apply_delta(delta):
        """Applies `delta` to `model.weights`."""
        tf.nest.assert_same_structure(delta, model.weights.trainable)

        grads_and_vars = []
        for e in delta.keys():
            if 'kernel_posterior_loc' in e:
                grads_and_vars.append((delta[e], model.weights.non_trainable[e.rsplit('/')[0] + '/s_loc']))
            if 'kernel_posterior_untransformed_scale' in e:
                grads_and_vars.append((delta[e], model.weights.non_trainable[e.rsplit('/')[0] + '/s_untrasformed_scale']))

        optimizer.apply_gradients(grads_and_vars, name='server_update')
        return tf.constant(1)  # We have to return something.

    weights_delta = tf.nest.map_structure(tf.zeros_like, model.weights.trainable)
    apply_delta(delta=weights_delta)
    optimizer_vars = optimizer.variables()

    return (apply_delta, optimizer_vars)
