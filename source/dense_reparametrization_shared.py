import tensorflow as tf
import tensorflow_probability as tfp
import math
from tfp_utils import renormalize_mean_field_normal_fn, default_tensor_multivariate_normal_fn, precision_from_scale, \
            compute_gaussian_ratio, compute_gaussian_prod, loc_ratio_from_precision
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers import util as tfp_layers_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers.recurrent import _caching_device
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops


inf = 1e15
softplus = tfp.bijectors.Softplus()
precision_from_untransformed_scale = tfp.bijectors.Chain([precision_from_scale, softplus])


class VariationalReparametrized(tf.keras.layers.Layer):

    def build_posterior_fn(self, shape, dtype, name, posterior_fn, prior_fn):
        self.non_trainable_variable_dict[name] = {}
        s_loc = self.add_variable(name=name+'_s_loc', shape=shape,
                                                               dtype=dtype, trainable=False,
                                                               initializer=tf.random_normal_initializer(
                                                                   stddev=0.1 * math.sqrt(self.num_clients)))
        scale_init = tf.random_normal_initializer(mean=+inf, stddev=0.1).__call__(shape=shape)
        scale_init = softplus.inverse(softplus.forward(scale_init) / math.sqrt(self.num_clients))
        s_untransformed_scale = self.add_variable(name=name+'_s_untransformed_scale', shape=shape,
                                                  dtype=dtype,
                                                  trainable=False,
                                                  initializer=tf.keras.initializers.constant(scale_init.numpy()))
        s_prec = tfp.util.DeferredTensor(s_untransformed_scale, precision_from_untransformed_scale)
        s_i_loc = self.add_variable(name=name+'_s_i_loc', shape=shape, dtype=dtype, trainable=False,
                                                               initializer=tf.random_normal_initializer(stddev=0.1))
        s_i_untransformed_scale = self.add_variable(name=name+'_s_i_untransformed_scale', shape=shape,
                                                    dtype=dtype,
                                                    trainable=False,
                                                    initializer=tf.random_normal_initializer(mean=+inf, stddev=0.1))
        s_i_prec = tfp.util.DeferredTensor(s_i_untransformed_scale, precision_from_untransformed_scale)
        prec_ratio = tfp.util.DeferredTensor(s_prec, lambda x: x - s_i_prec)

        def loc_reparametrization_fn(x):
            return loc_ratio_from_precision(x, s_prec, s_i_loc, s_i_prec)

        loc_ratio = tfp.util.DeferredTensor(s_loc, loc_reparametrization_fn)
        posterior_fn = posterior_fn(loc_ratio, prec_ratio)
        prior_fn = prior_fn(loc_ratio, prec_ratio, self.num_clients, self.prior_scale)

        self.non_trainable_variable_dict[name]['s'] = (s_loc, s_prec)
        self.non_trainable_variable_dict[name]['s_i'] = (s_i_loc, s_i_prec)
        return posterior_fn, prior_fn

    def compute_delta(self):
        delta_dict = {}
        for key in self.trainable_variable_dict.keys():
            delta_dict[key] = (self.delta_function(self.trainable_variable_dict[key],
                                                   self.non_trainable_variable_dict[key]['s_i']))
        return delta_dict

    def renew_s_i(self):
        for key in self.non_trainable_variable_dict.keys():
            self.non_trainable_variable_dict[key]['s_i'][0].assign(self.trainable_variable_dict[key][0])
            self.non_trainable_variable_dict[key]['s_i'][1].variables[0].assign(
                self.trainable_variable_dict[key][1].variables[0])

    def apply_delta(self, delta):
        for key in self.non_trainable_variable_dict.keys():
            loc, precision = self.apply_delta_function(self.non_trainable_variable_dict[key]['s'], delta[key])
            self.non_trainable_variable_dict[key]['s'][0].assign(loc)
            self.non_trainable_variable_dict[key]['s'][1].variables[0].assign(
                precision_from_untransformed_scale.inverse(precision))

    def receive_and_save_weights(self, layer_server):
        for key in self.non_trainable_variable_dict.keys():
            self.non_trainable_variable_dict[key]['s'][0].assign(layer_server.non_trainable_variable_dict[key]['s'][0])
            self.non_trainable_variable_dict[key]['s'][1].variables[0].assign(
                layer_server.non_trainable_variable_dict[key]['s'][1].variables[0])

    def initialize_kernel_posterior(self):
        for key in self.non_trainable_variable_dict.keys():
            self.trainable_variable_dict[key][0].assign(self.non_trainable_variable_dict[key]['s'][0])
            self.trainable_variable_dict[key][1].variables[0].assign(
                softplus.inverse(softplus.forward(
                    self.non_trainable_variable_dict[key]['s'][1].variables[0])*math.sqrt(self.num_clients)))

    def apply_damping(self, damping_factor):
        for key in self.non_trainable_variable_dict.keys():
            loc, prec = self.apply_delta_function((self.trainable_variable_dict[key][0],
                                                   self.trainable_variable_dict[key][1]*damping_factor),
                                                  (self.non_trainable_variable_dict[key]['s_i'][0],
                                                   self.non_trainable_variable_dict[key]['s_i'][1]*(1-damping_factor)))
            self.trainable_variable_dict[key][0].assign(loc)
            self.trainable_variable_dict[key][1].variables[0].assign(precision_from_untransformed_scale.inverse(prec))


class DenseReparametrizationShared(tfp.layers.DenseReparameterization, VariationalReparametrized):

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
        self.delta_function = lambda t1, t2: compute_gaussian_ratio(*t1, *t2)
        self.apply_delta_function = lambda t1, t2: compute_gaussian_prod(*t1, *t2)
        self.non_trainable_variable_dict = {}
        self.trainable_variable_dict = {}

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
        self.kernel_posterior_fn, self.kernel_prior_fn = self.build_posterior_fn(shape, dtype, name,
                                                                                 self.kernel_posterior_fn,
                                                                                 self.kernel_prior_fn)
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

        self.trainable_variable_dict['kernel'] = (self.kernel_posterior.distribution.loc.pretransformed_input,
                                                  self.kernel_posterior.distribution.scale
                                                  .pretransformed_input.pretransformed_input)
        self.built = True


class LSTMCellVariational(tf.keras.layers.LSTMCell, VariationalReparametrized):

    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 num_clients=1,
                 prior_scale=1.,
                 kernel_posterior_fn=renormalize_mean_field_normal_fn,
                 kernel_posterior_tensor_fn=(lambda d: d.sample()),
                 recurrent_kernel_posterior_fn=renormalize_mean_field_normal_fn,
                 recurrent_kernel_posterior_tensor_fn=(lambda d: d.sample()),
                 kernel_prior_fn=default_tensor_multivariate_normal_fn,
                 recurrent_kernel_prior_fn=default_tensor_multivariate_normal_fn,
                 kernel_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p)),
                 recurrent_kernel_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p)),
                 bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(
                     is_singular=True),
                 bias_posterior_tensor_fn=(lambda d: d.sample()),
                 bias_prior_fn=None,
                 bias_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p)),
                 **kwargs):

        super(LSTMCellVariational, self).__init__(units,
                                                  activation=activation,
                                                  recurrent_activation=recurrent_activation,
                                                  use_bias=use_bias,
                                                  kernel_initializer=kernel_initializer,
                                                  recurrent_initializer=recurrent_initializer,
                                                  bias_initializer=bias_initializer,
                                                  unit_forget_bias=unit_forget_bias,
                                                  kernel_regularizer=None,
                                                  recurrent_regularizer=None,
                                                  bias_regularizer=None,
                                                  kernel_constraint=kernel_constraint,
                                                  recurrent_constraint=recurrent_constraint,
                                                  bias_constraint=bias_constraint,
                                                  dropout=dropout,
                                                  recurrent_dropout=recurrent_dropout,
                                                  implementation=implementation,
                                                  **kwargs)
        self.kernel_posterior_fn = kernel_posterior_fn
        self.kernel_posterior_tensor_fn = kernel_posterior_tensor_fn
        self.recurrent_kernel_posterior_fn = recurrent_kernel_posterior_fn
        self.recurrent_kernel_posterior_tensor_fn = recurrent_kernel_posterior_tensor_fn
        self.kernel_posterior_tensor_fn = kernel_posterior_tensor_fn
        self.kernel_prior_fn = kernel_prior_fn
        self.recurrent_kernel_prior_fn = recurrent_kernel_prior_fn
        self.kernel_divergence_fn = kernel_divergence_fn
        self.recurrent_kernel_divergence_fn = recurrent_kernel_divergence_fn
        self.bias_posterior_fn = bias_posterior_fn
        self.bias_posterior_tensor_fn = bias_posterior_tensor_fn
        self.bias_prior_fn = bias_prior_fn
        self.bias_divergence_fn = bias_divergence_fn
        self.num_clients = num_clients
        self.prior_scale = prior_scale
        self.delta_function = lambda t1, t2: compute_gaussian_ratio(*t1, *t2)
        self.apply_delta_function = lambda t1, t2: compute_gaussian_prod(*t1, *t2)
        self.non_trainable_variable_dict = {}
        self.trainable_variable_dict = {}

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        input_dim = input_shape[-1]

        shape_kernel = (input_dim, self.units * 4)
        shape_recurrent = (self.units, self.units * 4)
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        self.kernel_posterior_fn, self.kernel_prior_fn = self.build_posterior_fn(shape_kernel, dtype, 'kernel',
                                                                                 self.kernel_posterior_fn,
                                                                                 self.kernel_prior_fn)

        self.recurrent_kernel_posterior_fn, self.recurrent_kernel_prior_fn = self.build_posterior_fn(shape_recurrent, dtype,
                                                                                                     'recurrent_kernel',
                                                                                 self.recurrent_kernel_posterior_fn,
                                                                                 self.recurrent_kernel_prior_fn)

        self.kernel_posterior = self.kernel_posterior_fn(dtype, shape_kernel, 'kernel_posterior', self.trainable,
                                                         self.add_variable,
                                                         initializer=self.kernel_initializer,
                                                         regularizer=self.kernel_regularizer,
                                                         constraint=self.kernel_constraint,
                                                         caching_device=default_caching_device)

        if self.kernel_prior_fn is None:
            self.kernel_prior = None
        else:
            self.kernel_prior = self.kernel_prior_fn(
                dtype, shape_kernel, 'kernel_prior',
                self.trainable, self.add_variable)

        self.recurrent_kernel_posterior = self.recurrent_kernel_posterior_fn(dtype, shape_recurrent, 'recurrent_kernel_posterior',
                                                                             self.trainable,
                                                                             self.add_variable,
                                                                             initializer=self.recurrent_initializer,
                                                                             regularizer=self.recurrent_regularizer,
                                                                             constraint=self.recurrent_constraint,
                                                                             caching_device=default_caching_device)

        if self.recurrent_kernel_prior_fn is None:
            self.recurrent_kernel_prior = None
        else:
            self.recurrent_kernel_prior = self.recurrent_kernel_prior_fn(
                dtype, shape_recurrent, 'recurrent_kernel_prior',
                self.trainable, self.add_variable)

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                        ])
            else:
                bias_initializer = self.bias_initializer

            self.bias = self.add_weight(
                          shape=(self.units * 4,),
                          name='bias',
                          initializer=bias_initializer,
                          regularizer=self.bias_regularizer,
                          constraint=self.bias_constraint,
                          caching_device=default_caching_device)
        else:
            self.bias = None


        self._apply_divergence(
            self.kernel_divergence_fn,
            self.kernel_posterior,
            self.kernel_prior,
            name='divergence_kernel')
        self._apply_divergence(
            self.recurrent_kernel_divergence_fn,
            self.recurrent_kernel_posterior,
            self.recurrent_kernel_prior,
            name='divergence_recurrent_kernel')

        self.trainable_variable_dict['kernel'] = (self.kernel_posterior.distribution.loc.pretransformed_input,
                                                  self.kernel_posterior.distribution.scale
                                                  .pretransformed_input.pretransformed_input)
        self.trainable_variable_dict['recurrent_kernel'] = (self.recurrent_kernel_posterior.distribution.loc.
                                                            pretransformed_input,
                                                            self.recurrent_kernel_posterior.distribution.scale
                                                            .pretransformed_input.pretransformed_input)
        self.built = True

    def _apply_divergence(self, divergence_fn, posterior, prior, name, posterior_tensor=None):
        divergence = tf.identity(
            divergence_fn(
                posterior, prior, posterior_tensor),
            name=name)
        self.add_loss(divergence)

    def sample_weights(self):
        self.kernel = self.kernel_posterior_tensor_fn(self.kernel_posterior)
        self.recurrent_kernel = self.recurrent_kernel_posterior_tensor_fn(self.recurrent_kernel_posterior)


class RNNVarReparametrized(tf.keras.layers.RNN):

    def compute_delta(self):
        return self.cell.compute_delta()

    def renew_s_i(self):
        self.cell.renew_s_i()

    def apply_delta(self, delta):
        self.cell.apply_delta(delta)

    def receive_and_save_weights(self, layer_server):
        self.cell.receive_and_save_weights(layer_server.cell)

    def initialize_kernel_posterior(self):
        self.cell.initialize_kernel_posterior()

    def apply_damping(self, damping_factor):
        self.cell.apply_damping(damping_factor)

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        self.cell.sample_weights()
        return super(RNNVarReparametrized, self).call(inputs,
                                                      mask=mask,
                                                      training=training,
                                                      initial_state=initial_state,
                                                      constants=constants)

