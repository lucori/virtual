import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
import math
from tensorflow_probability.python.layers import util as tfp_layers_util


def precision_from_scale(scale):
    return tf.math.reciprocal(tf.math.square(scale))


def scale_prod_from_precision(p1, p2):
    return tf.math.sqrt(tf.math.reciprocal(p1 + p2))


def scale_ratio_from_precision(p1, p2):
    return tf.math.sqrt(tf.math.reciprocal(p1 - p2))


def loc_prod(loc1, loc2, p1, p2, scale_prod):
    return tf.multiply(tf.math.square(scale_prod),
                       tf.math.multiply(loc1, p1) +
                       tf.math.multiply(loc2, p2))


def loc_ratio(loc1, loc2, p1, p2, scale_ratio):
    return tf.multiply(tf.math.square(scale_ratio),
                       tf.math.multiply(loc1, p1) -
                       tf.math.multiply(loc2, p2))


def compute_gaussian_prod(loc1, scale1, loc2, scale2):
    p1 = precision_from_scale(scale1)
    p2 = precision_from_scale(scale2)
    scale_prod = scale_prod_from_precision(p1, p2)
    loc_p = loc_prod(loc1, loc2, p1, p2, scale_prod)
    return loc_p, scale_prod


def compute_gaussian_ratio(loc1, scale1, loc2, scale2):
    p1 = precision_from_scale(scale1)
    p2 = precision_from_scale(scale2)
    scale_ratio = scale_ratio_from_precision(p1, p2)
    loc_r = loc_ratio(loc1, loc2, p1, p2, scale_ratio)
    return loc_r, scale_ratio


def renormalize_mean_field_normal_fn(loc_ratio, prec_ratio):
    loc_scale_fn = tensor_loc_scale_fn()

    def _fn(dtype, shape, name, trainable, add_variable_fn):
        loc, scale = loc_scale_fn(dtype, shape, name, trainable, add_variable_fn)
        if scale is None:
            dist = tfd.Deterministic(loc=loc)
        else:
            loc_reparametrized, scale_reparametrized = reparametrize_loc_scale(loc, scale, loc_ratio, prec_ratio)
            dist = tfd.Normal(loc=loc_reparametrized, scale=scale_reparametrized)

        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
    return _fn


def tensor_loc_scale_fn(is_singular=False, loc_initializer=tf.random_normal_initializer(stddev=0.1),
                        untransformed_scale_initializer=tf.random_normal_initializer(mean=-3., stddev=0.1),
                        loc_regularizer=None, untransformed_scale_regularizer=None, loc_constraint=None,
                        untransformed_scale_constraint=None):
    def _fn(dtype, shape, name, trainable, add_variable_fn):
        """Creates `loc`, `scale` parameters."""
        loc = add_variable_fn(
            name=name + '_loc',
            shape=shape,
            initializer=loc_initializer,
            regularizer=loc_regularizer,
            constraint=loc_constraint,
            dtype=dtype,
            trainable=trainable)
        if is_singular:
            return loc, None
        untransformed_scale = add_variable_fn(
            name=name + '_untransformed_scale',
            shape=shape,
            initializer=untransformed_scale_initializer,
            regularizer=untransformed_scale_regularizer,
            constraint=untransformed_scale_constraint,
            dtype=dtype,
            trainable=trainable)
        scale = tfp.util.DeferredTensor(untransformed_scale, tfp.bijectors.Softplus(), name=name + '_scale')
        return loc, scale
    return _fn


def reparametrize_loc_scale(loc, scale, loc_ratio, prec_ratio):
    prec = tfp.util.DeferredTensor(scale, precision_from_scale)

    def scale_reparametrization_fn(x):
        return scale_prod_from_precision(x, prec_ratio)

    scale_reparametrized = tfp.util.DeferredTensor(prec, scale_reparametrization_fn)

    def loc_reparametrization_fn(x):
        return loc_prod(x, loc_ratio, prec, prec_ratio, scale_reparametrized)

    loc_reparametrized = tfp.util.DeferredTensor(loc, loc_reparametrization_fn)
    return loc_reparametrized, scale_reparametrized


def default_tensor_multivariate_normal_fn(loc_ratio, prec_ratio, num_clients):
    def _fn(dtype, shape, name, trainable, add_variable_fn):
        del trainable
        loc_scale_fn = tensor_loc_scale_fn(loc_initializer=tf.keras.initializers.constant(0.),
                                           untransformed_scale_initializer=
                                           tf.keras.initializers.constant(math.sqrt(num_clients)))
        loc, scale = loc_scale_fn(dtype, shape, name, False, add_variable_fn)
        loc_reparametrized, scale_reparametrized = reparametrize_loc_scale(loc, scale, loc_ratio, prec_ratio)
        dist = tfd.Normal(loc=loc_reparametrized, scale=scale_reparametrized)
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
    return _fn


class DenseReparameterizationShared(tfp.layers.DenseReparameterization):

    def __init__(self, units,
                 activation=None,
                 activity_regularizer=None,
                 num_clients=1,
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

        super(DenseReparameterizationShared, self).__init__(units,
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
                                       initializer=tf.keras.initializers.constant(0.))
        s_untrasformed_scale = self.add_variable(name='s_untrasformed_scale', shape=[in_size, self.units], dtype=dtype,
                                                 trainable=False, initializer=tf.keras.initializers.constant(1.))
        self.s_scale = tfp.util.DeferredTensor(s_untrasformed_scale, tfp.bijectors.Softplus())
        self.s_i_loc = self.add_variable(name='s_i_loc', shape=[in_size, self.units], dtype=dtype, trainable=False,
                                       initializer=tf.keras.initializers.constant(0.))
        s_i_untrasformed_scale = self.add_variable(name='s_i_untrasformed_scale', shape=[in_size, self.units],
                                                   dtype=dtype,
                                                   trainable=False,
                                                   initializer=tf.keras.initializers.constant(math.sqrt(self.num_clients)))
        self.s_i_scale = tfp.util.DeferredTensor(s_i_untrasformed_scale, tfp.bijectors.Softplus())
        s_prec = tfp.util.DeferredTensor(self.s_scale, precision_from_scale)
        s_i_prec = tfp.util.DeferredTensor(self.s_i_scale, precision_from_scale)

        def scale_reparametrization_fn(x):
            return scale_ratio_from_precision(x, s_i_prec)

        scale_ratio = tfp.util.DeferredTensor(s_prec, scale_reparametrization_fn)
        self.prec_ratio = tfp.util.DeferredTensor(scale_ratio, precision_from_scale)

        def loc_reparametrization_fn(x):
            return loc_ratio(x, self.s_i_loc, s_prec, s_i_prec, scale_ratio)

        self.loc_ratio = tfp.util.DeferredTensor(self.s_loc, loc_reparametrization_fn)

        self.kernel_posterior_fn = self.kernel_posterior_fn(self.loc_ratio, self.prec_ratio)
        self.kernel_prior_fn = self.kernel_prior_fn(self.loc_ratio, self.prec_ratio, self.num_clients)

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


class BayesianSGD(tf.keras.optimizers.Optimizer):

    def __init__(self,
               learning_rate=0.01,
               name="BayesianSGD",
               **kwargs):
        super(BayesianSGD, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)

    def apply_gradients(self, grads_and_vars, **kwargs):

        for i, (grad, var) in enumerate(grads_and_vars):
            if 's_loc' in var.name:
                loc, scale = compute_gaussian_prod(var,
                                                   tfp.bijectors.Softplus().forward(grads_and_vars[i + 1][1]),
                                                   grad,
                                                   tfp.bijectors.Softplus().forward(grads_and_vars[i + 1][0])
                                                   )
                var.assign(loc)
                grads_and_vars[i + 1][1].assign(tfp.bijectors.Softplus().inverse(scale))

    def get_config(self):
        config = super(BayesianSGD, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
        })
        return config


