import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd
import numpy as np
from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import special_math
from tensorflow_probability.python.internal import tensor_util

eps = 1e-6


class NormalNatural(tfd.Distribution):

    def __init__(self,
                 gamma,
                 prec,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='NormalNatural'):
        """Construct Normal distributions with natural parameters `gamma` and `prec`.

        The parameters `gamma` and `prec` must be shaped in a way that supports
        broadcasting (e.g. `gamma + prec` is a valid operation).

        Args:
          gamma: Floating point tensor; the signal to noise ratio of the distribution(s).
          prec: Floating point tensor; the precision of the distribution(s).
            Must contain only positive values.
          validate_args: Python `bool`, default `False`. When `True` distribution
            parameters are checked for validity despite possibly degrading runtime
            performance. When `False` invalid inputs may silently render incorrect
            outputs.
          allow_nan_stats: Python `bool`, default `True`. When `True`,
            statistics (e.g., mean, mode, variance) use the value "`NaN`" to
            indicate the result is undefined. When `False`, an exception is raised
            if one or more of the statistic's batch members are undefined.
          name: Python `str` name prefixed to Ops created by this class.

        Raises:
          TypeError: if `gamma` and `prec` have different `dtype`.
        """
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([gamma, prec], dtype_hint=tf.float32)
            self._gamma = tensor_util.convert_nonref_to_tensor(
                gamma, dtype=dtype, name='gamma')
            self._prec = tensor_util.convert_nonref_to_tensor(
                prec, dtype=dtype, name='prec')
            super(NormalNatural, self).__init__(dtype=dtype,
                                                reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
                                                validate_args=validate_args,
                                                allow_nan_stats=allow_nan_stats,
                                                parameters=parameters,
                                                name=name)

    @staticmethod
    def _param_shapes(sample_shape):
        return dict(
            zip(('gamma', 'prec'),
                ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 2)))

    @classmethod
    def _params_event_ndims(cls):
        return dict(gamma=0, prec=0)

    @property
    def gamma(self):
        """Distribution parameter for the gamma."""
        return self._gamma

    @property
    def prec(self):
        """Distribution parameter for standard deviation."""
        return self._prec

    def _batch_shape_tensor(self, gamma=None, prec=None):
        return prefer_static.broadcast_shape(
            prefer_static.shape(self.gamma if gamma is None else gamma),
            prefer_static.shape(self.prec if prec is None else prec))

    def _batch_shape(self):
        return tf.broadcast_static_shape(self.gamma.shape, self.prec.shape)

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

    @property
    def loc(self):
        return tf.math.multiply_no_nan(self.gamma, tf.math.reciprocal_no_nan(self.prec))

    @property
    def scale(self):
        return tf.math.sqrt(tf.math.reciprocal_no_nan(self.prec))

    def _sample_n(self, n, seed=None):
        gamma = tf.convert_to_tensor(self.gamma)
        prec = tf.convert_to_tensor(self.prec)
        shape = tf.concat([[n], self._batch_shape_tensor(gamma=gamma, prec=prec)],
                          axis=0)
        sampled = tf.random.normal(
            shape=shape, mean=0., stddev=1., dtype=self.dtype, seed=seed)

        return tf.math.multiply(sampled, self.scale) + self.loc

    def _log_prob(self, x):
        prec = tf.convert_to_tensor(self.prec)
        log_unnormalized = - prec * tf.math.squared_difference(x, self.gamma)
        log_normalization = tf.constant(
            0.5 * np.log(2. * np.pi), dtype=self.dtype) - 0.5 * tf.math.log(prec)
        return log_unnormalized - log_normalization

    def _log_cdf(self, x):
        return special_math.log_ndtr(self._z(x))

    def _cdf(self, x):
        return special_math.ndtr(self._z(x))

    def _log_survival_function(self, x):
        return special_math.log_ndtr(-self._z(x))

    def _survival_function(self, x):
        return special_math.ndtr(-self._z(x))

    def _entropy(self):
        log_normalization = tf.constant(
            0.5 * np.log(2. * np.pi), dtype=self.dtype) - 0.5 * tf.math.log(self.prec)
        entropy = 0.5 + log_normalization
        return entropy * tf.ones_like(self.gamma)

    def _mean(self):
        return self.gamma / self.prec * tf.ones_like(self.prec)

    def _quantile(self, p):
        return special_math.ndtri(p) * self._stddev() + self._mean()

    def _stddev(self):
        return tf.math.sqrt(1. / self.prec) * tf.ones_like(self.gamma)

    _mode = _mean

    def _z(self, x, prec=None):
        """Standardize input `x` to a unit normal."""
        with tf.name_scope('standardize'):
            return (self.prec * x - self.gamma) / (tf.math.sqrt(self.prec) if prec is None else tf.math.sqrt(prec))

    def _default_event_space_bijector(self):
        return identity_bijector.Identity(validate_args=self.validate_args)

    def _parameter_control_dependencies(self, is_init):
        assertions = []

        if is_init:
            try:
                self._batch_shape()
            except ValueError:
                raise ValueError(
                    'Arguments `loc` and `scale` must have compatible shapes; '
                    'loc.shape={}, scale.shape={}.'.format(
                        self.gamma.shape, self.prec.shape))
            # We don't bother checking the shapes in the dynamic case because
            # all member functions access both arguments anyway.

        if not self.validate_args:
            assert not assertions  # Should never happen.
            return []

        if is_init != tensor_util.is_ref(self.scale):
            assertions.append(assert_util.assert_positive(
                self.scale, message='Argument `scale` must be positive.'))

        return assertions


@tfp.distributions.kullback_leibler.RegisterKL(NormalNatural, NormalNatural)
def _kl_normal_natural(a, b, name=None):
    """Calculate the batched KL divergence KL(a || b) with a and b Normal.

    Args:
      a: instance of a NormalNatural distribution object.
      b: instance of a NormalNatural distribution object.
      name: Name to use for created operations.
        Default value: `None` (i.e., `'kl_normal_natural'`).

    Returns:
      kl_div: Batchwise KL(a || b)
    """
    with tf.name_scope(name or 'kl_normal_natural'):
        a_prec = tf.convert_to_tensor(a.prec)
        b_prec = tf.convert_to_tensor(b.prec)  # We'll read it thrice.
        diff_log_prec = tf.math.log(a.prec + eps) - tf.math.log(b_prec + eps)
        inverse_a_prec = tf.math.reciprocal_no_nan(a_prec)
        inverse_b_prec = tf.math.reciprocal_no_nan(a_prec)
        return (
                0.5 * tf.multiply(b_prec, tf.math.squared_difference(tf.math.multiply(a.gamma, inverse_a_prec),
                                                                     tf.math.multiply(b.gamma, inverse_b_prec))) +
                0.5 * tf.math.expm1(- diff_log_prec) +
                0.5 * diff_log_prec)
