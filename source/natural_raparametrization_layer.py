import math
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers import util as tfp_layers_util


from source.centered_layers import LayerCentered
from source.tfp_utils import (renormalize_natural_mean_field_normal_fn,
                              natural_tensor_multivariate_normal_fn,
                              NaturalParTuple)


def natural_function(fun):
    def fn(t1, t2):
        result = []
        if issubclass(t1.__class__, tuple):
            for par1, par2 in zip(list(t1), list(t2)):
                result.append(fun(par1, par2))
            result = NaturalParTuple(result)
        else:
            result = fun(t1, t2)
        return result
    return fn


natural_ratio = natural_function(tf.subtract)
natural_prod = natural_function(tf.add)


class VariationalReparametrizedNatural(LayerCentered):

    def build_posterior_fn_natural(self, shape, dtype, name, posterior_fn, prior_fn):
        server_gamma = self.add_variable(name=name+'_server_gamma', shape=shape, dtype=dtype, trainable=False,
                                         initializer=tf.keras.initializers.zeros)
        server_prec = self.add_variable(name=name+'_server_precision', shape=shape, dtype=dtype, trainable=False,
                                        initializer=tf.keras.initializers.zeros)
        client_gamma = self.add_variable(name=name+'_client_gamma', shape=shape, dtype=dtype, trainable=False,
                                         initializer=tf.keras.initializers.zeros)
        client_prec = self.add_variable(name=name + '_client_precision', shape=shape, dtype=dtype, trainable=False,
                                         initializer=tf.keras.initializers.zeros)

        ratio_prec = tfp.util.DeferredTensor(server_prec, lambda x: x - client_prec)
        ratio_gamma = tfp.util.DeferredTensor(server_gamma, lambda x: x - client_gamma)

        posterior_fn = posterior_fn(ratio_gamma, ratio_prec)
        prior_fn = prior_fn(ratio_gamma, ratio_prec, self.num_clients, self.prior_scale)

        self.server_variable_dict[name] = NaturalParTuple((server_gamma, server_prec))
        self.client_center_variable_dict[name] = NaturalParTuple((client_gamma, client_prec))
        return posterior_fn, prior_fn

    def initialize_kernel_posterior(self):
        for key in self.client_variable_dict.keys():
            self.client_variable_dict[key].assign(self.server_variable_dict[key])

    def apply_damping(self, damping_factor):
        for key in self.server_variable_dict.keys():
            if issubclass(self.client_variable_dict[key].__class__, tuple):
                gamma, prec = self.apply_delta_function((self.client_variable_dict[key][0]*damping_factor,
                                                       self.client_variable_dict[key][1]*damping_factor),
                                                      (self.client_center_variable_dict[key][0]*(1-damping_factor),
                                                       self.client_center_variable_dict[key][1]*(1-damping_factor)))
                self.client_variable_dict[key].assign((gamma, prec))


class DenseSharedNatural(VariationalReparametrizedNatural):

    def __init__(self, units,
                 activation=None,
                 activity_regularizer=None,
                 num_clients=1,
                 prior_scale=1.,
                 trainable=True,
                 kernel_posterior_fn=renormalize_natural_mean_field_normal_fn,
                 kernel_posterior_tensor_fn=(lambda d: d.sample()),
                 kernel_prior_fn=natural_tensor_multivariate_normal_fn,
                 kernel_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p)),
                 bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(is_singular=True),
                 bias_posterior_tensor_fn=(lambda d: d.sample()),
                 bias_prior_fn=None,
                 bias_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p)),
                 **kwargs
                 ):

        self.precision_initializer = None
        if 'precision_initializer' in kwargs:
            self.precision_initializer = \
                kwargs.pop('precision_initializer')

        super(DenseSharedNatural, self).\
            __init__(units,
                     activation=activation,
                     activity_regularizer=activity_regularizer,
                     trainable=trainable,
                     kernel_posterior_fn=kernel_posterior_fn,
                     kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
                     kernel_prior_fn=kernel_prior_fn,
                     kernel_divergence_fn=kernel_divergence_fn,
                     bias_posterior_fn=bias_posterior_fn,
                     bias_posterior_tensor_fn=bias_posterior_tensor_fn,
                     bias_prior_fn=bias_prior_fn,
                     bias_divergence_fn=bias_divergence_fn,
                     **kwargs)

        self.num_clients = num_clients
        self.prior_scale = prior_scale
        self.delta_function = natural_ratio
        self.apply_delta_function = natural_prod
        self.client_variable_dict = {}
        self.client_center_variable_dict = {}
        self.server_variable_dict = {}

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        in_size = tf.compat.dimension_value(input_shape.with_rank_at_least(2)[-1])
        if in_size is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self._input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: in_size})

        # If self.dtype is None, build weights using the default dtype.
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        shape = [in_size, self.units]
        name = 'kernel'
        self.kernel_posterior_fn, self.kernel_prior_fn = \
            self.build_posterior_fn_natural(shape, dtype, name,
                                            self.kernel_posterior_fn,
                                            self.kernel_prior_fn)
        if self.precision_initializer:
            self.kernel_posterior = self.kernel_posterior_fn(
                dtype, [in_size, self.units], 'kernel_posterior',
                self.trainable, self.add_variable,
                precision_initializer=self.precision_initializer)
        else:
            self.kernel_posterior = self.kernel_posterior_fn(
                dtype, [in_size, self.units], 'kernel_posterior',
                self.trainable, self.add_variable)

        if self.kernel_prior_fn is None:
            self.kernel_prior = None
        else:
            self.kernel_prior = self.kernel_prior_fn(
                dtype, [in_size, self.units], 'kernel_prior',
                self.trainable, self.add_variable)

        if self.bias_posterior_fn is None:
            self.bias_posterior = None
        else:
            self.bias_posterior = self.bias_posterior_fn(
                dtype, [self.units], 'bias_posterior',
                self.trainable, self.add_variable)

        if self.bias_prior_fn is None:
            self.bias_prior = None
        else:
            self.bias_prior = self.bias_prior_fn(
                dtype, [self.units], 'bias_prior',
                self.trainable, self.add_variable)

        self.client_variable_dict['kernel'] = NaturalParTuple((
            self.kernel_posterior.distribution.gamma.pretransformed_input,
            self.kernel_posterior.distribution.prec.pretransformed_input))

        self.bias_center = self.add_weight('bias_center',
                                           shape=[self.units, ],
                                           initializer=tf.keras.initializers.constant(0.),
                                           dtype=self.dtype,
                                           trainable=False)
        self.client_variable_dict['bias'] = self.bias_posterior.distribution.loc
        self.server_variable_dict['bias'] = self.bias_posterior.distribution.loc
        self.client_center_variable_dict['bias'] = self.bias_center
        self.built = True


class DenseReparametrizationNaturalShared(DenseSharedNatural, tfp.layers.DenseReparameterization):
    pass