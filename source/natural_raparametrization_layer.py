import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers import util as tfp_layers_util
from source.centered_layers import LayerCentered
from source.tfp_utils import precision_from_untransformed_scale, precision_from_scale
from source.normal_natural import NormalNatural, eps
from tensorflow.python.keras.constraints import Constraint
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops


class NonNegPrec(Constraint):

    def __call__(self, w):
        prec = w[..., -1]
        prec = prec * math_ops.cast(math_ops.greater_equal(prec, eps), K.floatx())
        return tf.stack([w[..., 0], prec], axis=-1)


class VariationalReparametrizedNatural(LayerCentered):

    def build_posterior_fn_natural(self, shape, dtype, name, posterior_fn, prior_fn):
        natural_par_shape = shape + [2]
        server_par = self.add_variable(name=name+'_server_par', shape=natural_par_shape, dtype=dtype, trainable=False,
                                       initializer=tf.keras.initializers.zeros)
        client_par = self.add_variable(name=name+'_client_par', shape=natural_par_shape, dtype=dtype, trainable=False,
                                       initializer=tf.keras.initializers.zeros)

        ratio_par = tfp.util.DeferredTensor(server_par, lambda x: x - client_par)

        posterior_fn = posterior_fn(ratio_par)
        prior_fn = prior_fn(ratio_par, self.num_clients, self.prior_scale)

        self.server_variable_dict[name] = server_par
        self.client_center_variable_dict[name] = client_par
        return posterior_fn, prior_fn

    def initialize_kernel_posterior(self):
        for key in self.client_variable_dict.keys():
            self.client_variable_dict[key].assign(self.server_variable_dict[key])

    def apply_damping(self, damping_factor):
        for key in self.server_variable_dict.keys():
            tf.debugging.check_numerics(self.client_variable_dict[key], 'client')
            tf.debugging.check_numerics(self.client_center_variable_dict[key], 'center')
            damped = self.apply_delta_function(self.client_variable_dict[key] * damping_factor,
                                               self.client_center_variable_dict[key] * (1-damping_factor))
            self.client_variable_dict[key].assign(damped)

    def renormalize_natural_mean_field_normal_fn(self, ratio_par):

        def _fn(dtype, shape, name, trainable, add_variable_fn,
                natural_initializer=None,
                natural_regularizer=None, natural_constraint=NonNegPrec(),
                **kwargs):
            natural_par_fn = self.tensor_natural_par_fn(natural_initializer=natural_initializer,
                                                        natural_regularizer=natural_regularizer,
                                                        natural_constraint=natural_constraint,
                                                        **kwargs)
            natural = natural_par_fn(dtype, shape, name, trainable, add_variable_fn)
            self.client_variable_dict['kernel'] = natural
            natural_reparametrized = tfp.util.DeferredTensor(natural, lambda x: tf.add(x, ratio_par))
            gamma = tfp.util.DeferredTensor(natural_reparametrized, lambda x: x[..., 0], shape=shape)
            prec = tfp.util.DeferredTensor(natural_reparametrized, lambda x: x[..., 1], shape=shape)

            #scale = tfp.util.DeferredTensor(prec, precision_from_scale.inverse)
            #loc = tfp.util.DeferredTensor(gamma, lambda x: tf.math.xdivy(x, prec))
            dist = NormalNatural(gamma=gamma, prec=prec)
            #dist = tfd.Normal(loc=loc, scale=scale)
            batch_ndims = tf.size(dist.batch_shape_tensor())
            return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

        return _fn

    def tensor_natural_par_fn(self, is_singular=False, natural_initializer=tf.constant_initializer(0.),
                              natural_regularizer=None, natural_constraint=None,
                              **kwargs):
        def _fn(dtype, shape, name, trainable, add_variable_fn):
            """Creates 'natural' parameters."""
            natural = add_variable_fn(
                name=name + '_natural',
                shape=shape + [2],
                initializer=natural_initializer,
                regularizer=natural_regularizer,
                constraint=natural_constraint,
                dtype=dtype,
                trainable=trainable,
                **kwargs)
            return natural

        return _fn

    def natural_tensor_multivariate_normal_fn(self, ratio_par, num_clients, prior_scale=1.):
        def _fn(dtype, shape, name, trainable, add_variable_fn, initializer=natural_prior_initializer_fn(),
                regularizer=None, constraint=None, **kwargs):
            del trainable
            natural_par_fn = self.tensor_natural_par_fn(natural_initializer=initializer,
                                                        natural_regularizer=regularizer,
                                                        natural_constraint=constraint,
                                                        **kwargs)
            natural = natural_par_fn(dtype, shape, name, False, add_variable_fn)
            natural_reparametrized = tfp.util.DeferredTensor(natural, lambda x: x + ratio_par)
            gamma = tfp.util.DeferredTensor(natural_reparametrized, lambda x: x[..., 0], shape=shape)
            prec = tfp.util.DeferredTensor(natural_reparametrized, lambda x: x[..., 1], shape=shape)
            #TODO: we should use the num_clients to scale the prior

            #scale = tfp.util.DeferredTensor(prec, precision_from_scale.inverse)
            #loc = tfp.util.DeferredTensor(gamma, lambda x: tf.math.xdivy(x, prec))
            dist = NormalNatural(gamma=gamma, prec=prec)
            #dist = tfd.Normal(loc=loc, scale=scale)
            batch_ndims = tf.size(input=dist.batch_shape_tensor())
            return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

        return _fn

    def check_weights(self):
        for weight in self.get_weights():
            tf.debugging.check_numerics(weight, 'weight')


class DenseSharedNatural(VariationalReparametrizedNatural):

    def __init__(self, units,
                 activation=None,
                 activity_regularizer=None,
                 num_clients=1,
                 prior_scale=1.,
                 trainable=True,
                 kernel_posterior_fn=None,
                 kernel_posterior_tensor_fn=(lambda d: d.sample()),
                 kernel_prior_fn=None,
                 kernel_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p)),
                 bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(is_singular=True),
                 bias_posterior_tensor_fn=(lambda d: d.sample()),
                 bias_prior_fn=None,
                 bias_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p)),
                 **kwargs
                 ):

        self.untransformed_scale_initializer = None
        if 'untransformed_scale_initializer' in kwargs:
            self.untransformed_scale_initializer = \
                kwargs.pop('untransformed_scale_initializer')

        if kernel_posterior_fn is None:
            kernel_posterior_fn = self.renormalize_natural_mean_field_normal_fn
        if kernel_prior_fn is None:
            kernel_prior_fn = self.natural_tensor_multivariate_normal_fn

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
        self.delta_function = tf.subtract
        self.apply_delta_function = tf.add
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

        natural_initializer = natural_initializer_fn(loc_stdev=0.1, u_scale_init_avg=-5, u_scale_init_stdev=0.1,
                                                     untransformed_scale_initializer=self.untransformed_scale_initializer)

        self.kernel_posterior = self.kernel_posterior_fn(
                dtype, [in_size, self.units], 'kernel_posterior',
                self.trainable, self.add_variable,
                natural_initializer=natural_initializer)

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

        if self.bias_posterior:
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


class DenseLocalReparametrizationNaturalShared(DenseSharedNatural, tfp.layers.DenseLocalReparameterization):
    def _apply_variational_kernel(self, inputs):
        self.kernel_posterior_affine = tfd.Normal(
            loc=tf.matmul(inputs, self.kernel_posterior.distribution.loc),
            scale=tf.sqrt(tf.matmul(tf.math.square(inputs), tf.math.square(self.kernel_posterior.distribution.scale))))
        self.kernel_posterior_affine_tensor = (
            self.kernel_posterior_tensor_fn(self.kernel_posterior_affine))
        self.kernel_posterior_tensor = None
        return self.kernel_posterior_affine_tensor


def natural_initializer_fn(loc_stdev=0.1, u_scale_init_avg=-5, u_scale_init_stdev=0.1,
                           untransformed_scale_initializer=None):
    loc_init = tf.random_normal_initializer(stddev=loc_stdev)
    if untransformed_scale_initializer is None:
        untransformed_scale_initializer = tf.random_normal_initializer(mean=u_scale_init_avg, stddev=u_scale_init_stdev)

    def natural_initializer(shape, dtype=tf.float32):
        prec = precision_from_untransformed_scale(untransformed_scale_initializer(shape[:-1], dtype))
        gamma = loc_init(shape[:-1], dtype) * prec
        natural = tf.stack([gamma, prec], axis=-1)
        tf.debugging.check_numerics(natural, 'initializer')
        return natural

    return natural_initializer


def natural_prior_initializer_fn():
    gamma_init = tf.constant_initializer(0.)
    precision_init = tf.constant_initializer(1.)

    def natural_initializer(shape, dtype):
        prec = precision_init(shape[:-1], dtype)
        gamma = gamma_init(shape[:-1], dtype)
        natural = tf.stack([gamma, prec], axis=-1)
        return natural

    return natural_initializer
