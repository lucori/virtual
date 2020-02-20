import tensorflow as tf
import tensorflow_probability as tfp
import math
from tfp_utils import renormalize_mean_field_normal_fn, default_tensor_multivariate_normal_fn, precision_from_scale, \
            compute_gaussian_ratio, compute_gaussian_prod, loc_ratio_from_precision
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers import util as tfp_layers_util


inf = 1e15
softplus = tfp.bijectors.Softplus()
precision_from_untransformed_scale = tfp.bijectors.Chain([precision_from_scale, softplus])


class DenseReparametrizationShared(tfp.layers.DenseReparameterization):

    def __init__(self, units,
                 activation=None,
                 activity_regularizer=None,
                 num_clients=1,
                 prior_scale=1.,
                 trainable=True,
                 kernel_posterior_fn=renormalize_mean_field_normal_fn,
                 kernel_posterior_tensor_fn=(lambda d: d.sample()),
                 kernel_prior_fn=default_tensor_multivariate_normal_fn,
                 kernel_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p)),
                 bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(is_singular=True),
                 bias_posterior_tensor_fn=(lambda d: d.sample()),
                 bias_prior_fn=None,
                 bias_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p)),
                 **kwargs
                 ):

        super(DenseReparametrizationShared, self).__init__(units,
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
        print(self.num_clients, self.prior_scale)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        in_size = tf.compat.dimension_value(input_shape.with_rank_at_least(2)[-1])
        if in_size is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self._input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: in_size})

        # If self.dtype is None, build weights using the default dtype.
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

        self.s_loc = self.add_variable(name='s_loc', shape=[in_size, self.units], dtype=dtype, trainable=False,
                                       initializer=tf.random_normal_initializer(stddev=0.1*math.sqrt(self.num_clients)))
        scale_init = tf.random_normal_initializer(mean=+inf, stddev=0.1).__call__(shape=[in_size, self.units])
        scale_init = softplus.inverse(softplus.forward(scale_init)/math.sqrt(self.num_clients))
        s_untransformed_scale = self.add_variable(name='s_untransformed_scale', shape=[in_size, self.units], dtype=dtype,
                                                  trainable=False,
                                                  initializer=tf.keras.initializers.constant(scale_init.numpy()))
        self.s_prec = tfp.util.DeferredTensor(s_untransformed_scale, precision_from_untransformed_scale)

        self.s_i_loc = self.add_variable(name='s_i_loc', shape=[in_size, self.units], dtype=dtype, trainable=False,
                                         initializer=tf.random_normal_initializer(stddev=0.1))
        s_i_untransformed_scale = self.add_variable(name='s_i_untransformed_scale', shape=[in_size, self.units],
                                                    dtype=dtype,
                                                    trainable=False,
                                                    initializer=tf.random_normal_initializer(mean=+inf, stddev=0.1))
        self.s_i_prec = tfp.util.DeferredTensor(s_i_untransformed_scale, precision_from_untransformed_scale)
        self.prec_ratio = tfp.util.DeferredTensor(self.s_prec, lambda x: x - self.s_i_prec)

        def loc_reparametrization_fn(x):
            return loc_ratio_from_precision(x, self.s_prec, self.s_i_loc, self.s_i_prec)

        self.loc_ratio = tfp.util.DeferredTensor(self.s_loc, loc_reparametrization_fn)
        self.kernel_posterior_fn = self.kernel_posterior_fn(self.loc_ratio, self.prec_ratio)
        self.kernel_prior_fn = self.kernel_prior_fn(self.loc_ratio, self.prec_ratio, self.num_clients, self.prior_scale)

        # Must have a posterior kernel.
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

        self.built = True

    def compute_delta(self):
        loc, precision = compute_gaussian_ratio(self.variables[0],
                                                precision_from_untransformed_scale.forward(self.variables[1]),
                                                self.s_i_loc, self.s_i_prec)
        return loc, precision

    def renew_s_i(self):
        self.s_i_loc.assign(self.variables[0])
        self.s_i_prec.variables[0].assign(self.variables[1])

    def apply_delta(self, delta):
        loc, precision = compute_gaussian_prod(self.s_loc, self.s_prec, *delta)
        self.s_loc.assign(loc)
        self.s_prec.variables[0].assign(precision_from_untransformed_scale.inverse(precision))

    def receive_and_save_weights(self, layer_server):
        self.s_loc.assign(layer_server.variables[3])
        self.s_prec.variables[0].assign(layer_server.variables[4])

    def initialize_kernel_posterior(self):
        self.variables[0].assign(self.variables[3])
        self.variables[1].assign(softplus.inverse(softplus.forward(self.variables[4])*math.sqrt(self.num_clients)))

    def apply_damping(self, damping_factor):
        loc, prec = compute_gaussian_prod(self.variables[0], precision_from_scale(softplus.forward(self.variables[1])/
                                                                                  math.sqrt(damping_factor)),
                                          self.variables[3], precision_from_scale(softplus.forward(self.variables[4])/
                                                                                  math.sqrt(1-damping_factor)))
        self.variables[0].assign(loc)
        self.variables[1].assign(precision_from_untransformed_scale.inverse(prec))

