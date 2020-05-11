import math

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python import distributions as tfd


eps = 1/tf.float32.max
softplus = tfp.bijectors.Softplus()
precision_from_scale = tfp.bijectors.Chain([tfp.bijectors.Reciprocal(), tfp.bijectors.Square()])
precision_from_untransformed_scale = tfp.bijectors.Chain([precision_from_scale, softplus])


def loc_prod_from_locprec(loc_times_prec, sum_prec):
    #TODO: check if condition tf.abs(sum_prec) > eps is ever satisfied and if the formula is stable
    #TODO: in particular what if also tf.abs(sum_prec) < eps
    return tf.where(tf.abs(sum_prec) > eps, tf.math.xdivy(loc_times_prec, sum_prec),
                    tf.float32.max*tf.sign(loc_times_prec)*tf.sign(sum_prec))


def loc_prod_from_precision(loc1, p1, loc2, p2):
    prec_prod = p1 + p2
    loc1p1 = tf.math.multiply(loc1, p1)
    loc2p2 = tf.math.multiply(loc2, p2)
    return loc_prod_from_locprec(loc1p1 + loc2p2, prec_prod)


def compute_gaussian_prod(loc1, p1, loc2, p2):
    loc_prod = loc_prod_from_precision(loc1, p1, loc2, p2)
    return loc_prod, p1 + p2


def loc_ratio_from_precision(loc1, p1, loc2, p2):
    return loc_prod_from_precision(loc1, p1, loc2, -p2)


def compute_gaussian_ratio(loc1, p1, loc2, p2):
    return compute_gaussian_prod(loc1, p1, loc2, -p2)


def renormalize_mean_field_normal_fn(loc_ratio, prec_ratio):

    def _fn(dtype, shape, name, trainable, add_variable_fn,
            initializer=tf.random_normal_initializer(stddev=0.1),
            regularizer=None, constraint=None, **kwargs):
        loc_scale_fn = tensor_loc_scale_fn(loc_initializer=initializer,
                                           loc_regularizer=regularizer,
                                           loc_constraint=constraint, **kwargs)

        loc, scale = loc_scale_fn(dtype, shape, name, trainable, add_variable_fn)
        prec = tfp.util.DeferredTensor(scale, precision_from_scale, name='precision')
        if scale is None:
            dist = tfd.Deterministic(loc=loc)
        else:
            loc_reparametrized, scale_reparametrized = reparametrize_loc_scale(loc, prec, loc_ratio, prec_ratio)
            dist = tfd.Normal(loc=loc_reparametrized, scale=scale_reparametrized)

        batch_ndims = tf.size(dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
    return _fn


def default_tensor_multivariate_normal_fn(loc_ratio, prec_ratio, num_clients, prior_scale=1.):
    def _fn(dtype, shape, name, trainable, add_variable_fn, initializer=tf.keras.initializers.constant(0.),
            regularizer=None, constraint=None, **kwargs):
        del trainable
        loc_scale_fn = tensor_loc_scale_fn(loc_initializer=initializer,
                                           loc_regularizer=regularizer,
                                           loc_constraint=constraint,
                                           untransformed_scale_initializer=tf.keras.initializers.constant(
                                           tfp.bijectors.Softplus().inverse(prior_scale*math.sqrt(num_clients)).numpy()),
                                           **kwargs)
        loc, scale = loc_scale_fn(dtype, shape, name, False, add_variable_fn)
        prec = tfp.util.DeferredTensor(scale, precision_from_scale)
        loc_reparametrized, scale_reparametrized = reparametrize_loc_scale(loc, prec, loc_ratio, prec_ratio)
        dist = tfd.Normal(loc=loc_reparametrized, scale=scale_reparametrized)
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
    return _fn


def tensor_loc_scale_fn(is_singular=False,
                        loc_initializer
                        =tf.random_normal_initializer(stddev=0.1),

                        untransformed_scale_initializer
                        =tf.random_normal_initializer(mean=-3., stddev=0.1),

                        loc_regularizer=None,
                        untransformed_scale_regularizer=None,
                        loc_constraint=None,
                        untransformed_scale_constraint=None,
                        **kwargs):
    def _fn(dtype, shape, name, trainable, add_variable_fn):
        """Creates `loc`, `scale` parameters."""
        loc = add_variable_fn(
            name=name + '_loc',
            shape=shape,
            initializer=loc_initializer,
            regularizer=loc_regularizer,
            constraint=loc_constraint,
            dtype=dtype,
            trainable=trainable,
            **kwargs)
        if is_singular:
            return loc, None
        untransformed_scale = add_variable_fn(
            name=name + '_untransformed_scale',
            shape=shape,
            initializer=untransformed_scale_initializer,
            regularizer=untransformed_scale_regularizer,
            constraint=untransformed_scale_constraint,
            dtype=dtype,
            trainable=trainable,
            **kwargs)
        scale = tfp.util.DeferredTensor(untransformed_scale, tfp.bijectors.Softplus(), name=name + '_scale')
        return loc, scale
    return _fn


def reparametrize_loc_scale(loc, prec, loc_ratio, prec_ratio):
    precision_reparametrized = tfp.util.DeferredTensor(prec, lambda x: x + prec_ratio)

    def loc_reparametrization_fn(x):
        return loc_prod_from_precision(x, prec, loc_ratio, prec_ratio)

    loc_reparametrized = tfp.util.DeferredTensor(loc, loc_reparametrization_fn)
    scale_reparametrized = tfp.util.DeferredTensor(precision_reparametrized, precision_from_scale.inverse)
    return loc_reparametrized, scale_reparametrized

#TODO: reparametrize everything using natural parameter of gaussian
#TODO: implement natural parameter gaussian distribution


class LocPrecTuple(tuple):

    def assign(self, loc_prec_tuple):
        self[0].assign(loc_prec_tuple[0])
        self[1].variables[0].assign(precision_from_untransformed_scale.inverse(loc_prec_tuple[1]))
