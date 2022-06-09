import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.python.keras.layers.pooling
from tensorflow.keras.constraints import *
from tensorflow.keras.initializers import *
from keras.layers.convolutional.base_depthwise_conv import DepthwiseConv

import operator
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.layers.pooling import Pooling2D
from keras.layers.convolutional.base_conv import Conv
from keras.utils import tf_utils

import functools
import numpy as np

########################################################################################################################
EPSILON = tf.keras.backend.epsilon()
NEW = tf.newaxis
########################################################################################################################

########################################################################################################################
class AdaPool(Pooling2D):
    """
    AdaPool: Exponential Adaptive Pooling for Information-Retaining Downsampling
        https://arxiv.org/abs/2111.00772
    """
    def __init__(self, pool_size=2, strides=2, padding='VALID', data_format=None, name=None, **kwargs):
        super(AdaPool, self).__init__(
            self.pool_function, pool_size=pool_size, strides=strides, padding=padding, name=name,
            data_format=data_format, **kwargs
        )
    def build(self, input_shape):
        self.n_ch = input_shape[-1]
        self.beta = self.add_weight(
            "beta", shape=[1], initializer=Constant(0.5), constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
        )
    def eMPool(self, x, axis=-1):
        x *= tf.nn.softmax(x, axis)
        return tf.reduce_sum(x, axis)

    def eDSCWPool(self, x, axis=-1):
        DSC = lambda x, x_: tf.math.abs(2 * (x * x_)) / (x ** 2 + x_ ** 2 + EPSILON)
        x_ = tf.reduce_mean(x, axis, keepdims=True)
        dsc = tf.math.exp(DSC(x, x_))
        output = dsc * x / tf.reduce_sum(dsc, axis, keepdims=True)
        return tf.reduce_sum(output, axis)

    def pool_function(self, input, ksize, strides, padding, data_format=None):
        if data_format == 'channels_first':
            input = tf.transpose(input, [0, 2, 3, 1])
        patches = tf.image.extract_patches(input, ksize, strides, [1, 1, 1, 1], padding)
        # patches = tf.stack(tf.split(patches, patches.shape[-1] // self.n_ch, -1), -1)
        return self.eMPool(patches) * self.beta + self.eDSCWPool(patches) * (1 - self.beta)

########################################################################################################################
class SaBN(Layer):
    """
    Sandwich Batch Normalization: A Drop-In Replacement for Feature Distribution Heterogeneity
        https://arxiv.org/abs/2102.11382
    """
    def __init__(self, n_class, axis=-1):
        super(SaBN, self).__init__()
        self.n_class = n_class
        self.BN = BatchNormalization(axis=axis)
        self.axis = [axis] if isinstance(axis, int) else axis

    def build(self, input_shape):
        param_shape = self.get_param_shape(input_shape)

        self.scale = self.add_weight("scale", shape=param_shape, initializer='ones')
        self.offset = self.add_weight("offset", shape=param_shape, initializer='zeros')

    def get_param_shape(self, input_shape):
        ndims = len(input_shape)
        # negative parameter to positive parameter
        axis = [ndims + ax if ax < 0 else ax for ax in self.axis]
        axis_to_dim = {x: input_shape[x] for x in axis}
        param_shape = [axis_to_dim[i] if i in axis_to_dim else 1 for i in range(ndims)]
        param_shape = [self.n_class] + param_shape
        print(f'param_shape = {param_shape}')
        return param_shape

    def get_slice(self, x, label):
        x = tf.gather_nd(x, label)
        x = tf.squeeze(x, 1)

        return x

    def call(self, inputs, label, training=None, **kargs):
        if training == False:
            return self.BN(inputs, training=training)
        output = self.BN(inputs, training=training)

        # label = tf.argmax(label, -1)
        scale = self.get_slice(self.scale, label)
        offset = self.get_slice(self.offset, label)
        print(f'label = {label}')
        print(f'offset = {self.offset.shape}')
        print(f'offset_gather = {offset.shape}')
        output = scale * output + offset
        print(f'output = {output.shape}')
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_class': self.n_class,
        })
        return config

########################################################################################################################
class DepthwiseConv3D(DepthwiseConv):
    def __init__(self,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='same',
                 depth_multiplier=1,
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv3D, self).__init__(
            3,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def call(self, inputs):
        outputs = backend.conv3d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.use_bias:
            outputs = backend.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0],
                                             self.dilation_rate[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1],
                                             self.dilation_rate[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

########################################################################################################################
class global_attention(Layer):
    """
    Global Attention Mechanism: Retain Information to Enhance Channel-Spatial Interactions
        https://arxiv.org/abs/2112.05561
    :return:
    """
    def __init__(self, kernel=7, groups=1, squeeze_rate=0.7):
        super(global_attention, self).__init__()
        self.squeeze_rate = squeeze_rate
        self.kernel = kernel
        self.groups = groups

    def build(self, input_shape):
        rank = tf.rank(input_shape)
        self.GAP = GlobalAveragePooling2D if rank == 4 else GlobalAveragePooling3D
        self.GMP = GlobalMaxPooling2D if rank == 4 else GlobalMaxPooling3D

        self.activation = tf.nn.relu

        self.squeeze = Dense(int(input_shape[-1] * self.squeeze_rate))
        self.extend = Dense(input_shape[-1], activation='relu')

        conv = Conv2D if rank == 4 else Conv3D
        self.spatial_convolution = conv(1, self.kernel, padding='same', groups=self.groups)

    def channel_attention(self, x):
        def main(x):
            x = self.squeeze(x)
            x = self.extend(x)
            x = self.activation(x)
            return x
        GAP = main(self.GAP()(x))
        GMP = main(self.GMP()(x))

        return tf.nn.sigmoid(GAP + GMP)

    def spatial_attention(self, x):
        GAP = tf.reduce_mean(x[0], -1)
        GMP = tf.reduce_max(x[1], -1)

        spatial_atttention = self.spatial_convolution(tf.concat([GAP, GMP], -1))
        spatial_atttention = tf.nn.sigmoid(spatial_atttention)
        return spatial_atttention

    def call(self, inputs, label=0, training=None):
        channel_attention = self.channel_attention(inputs)
        x = inputs * channel_attention

        spatial_attention = self.spatial_attention(x)
        x *= spatial_attention

        return x + inputs

########################################################################################################################
# class LI_2D(tf.python.keras.layers.convolutional.Conv2D):
#     """
#     Dilated Convolutions with Lateral Inhibitions for Semantic Image Segmentation
#         https://arxiv.org/abs/2006.03708
#     """
#     def __init__(self,
#                  filters,
#                  kernel_size=3,
#                  strides=1,
#                  padding='valid',
#                  data_format=None,
#                  dilation_rate=1,
#                  groups=1,
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer='zeros',
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  intensity=0.2,
#                  **kwargs):
#         super(LI_2D, self).__init__(
#             filters=filters,
#             kernel_size=kernel_size,
#             strides=strides,
#             padding=padding,
#             data_format=data_format,
#             dilation_rate=dilation_rate,
#             groups=groups,
#             activation=activations.get(activation),
#             use_bias=use_bias,
#             kernel_initializer=initializers.get(kernel_initializer),
#             bias_initializer=initializers.get(bias_initializer),
#             kernel_regularizer=regularizers.get(kernel_regularizer),
#             bias_regularizer=regularizers.get(bias_regularizer),
#             activity_regularizer=regularizers.get(activity_regularizer),
#             kernel_constraint=constraints.get(kernel_constraint),
#             bias_constraint=constraints.get(bias_constraint),
#             **kwargs
#         )
#         self.intensity = intensity
#
#     def build(self, input_shape):
#         input_shape = tensor_shape.TensorShape(input_shape)
#         if len(input_shape) != 4:
#             raise ValueError('Inputs should have rank 4. Received input '
#                              'shape: ' + str(input_shape))
#         input_channel = self._get_input_channel(input_shape)
#         if input_channel % self.groups != 0:
#             raise ValueError(
#                 'The number of input channels must be evenly divisible by the number '
#                 'of groups. Received groups={}, but the input has {} channels '
#                 '(full input shape is {}).'.format(self.groups, input_channel,
#                                                    input_shape))
#         kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)
#
#         self.weight = self.add_weight("weight", shape=input_channel, initializer='HeNormal')
#
#         channel_axis = self._get_channel_axis()
#         if input_shape.dims[channel_axis].value is None:
#             raise ValueError('The channel dimension of the inputs '
#                              'should be defined. Found `None`.')
#         input_dim = int(input_shape[channel_axis])
#         self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
#
#         self.kernel = self.add_weight(
#             name='kernel',
#             shape=kernel_shape,
#             initializer=self.kernel_initializer,
#             regularizer=self.kernel_regularizer,
#             constraint=self.kernel_constraint,
#             trainable=True,
#             dtype=self.dtype)
#         if self.use_bias:
#             self.bias = self.add_weight(
#                 name='bias',
#                 shape=(self.filters,),
#                 initializer=self.bias_initializer,
#                 regularizer=self.bias_regularizer,
#                 constraint=self.bias_constraint,
#                 trainable=True,
#                 dtype=self.dtype)
#         else:
#             self.bias = None
#         self.built = True
#
#     def distance_factor(self, size=3, factor=1.5):
#         """ Euclidean distance """
#         x = np.mgrid[:size, :size]
#         x = (x - size // 2) * 1.0
#         x = (x**2).sum(0) + factor
#         x = tf.Variable(1/x, dtype='float32')
#         return x
#
#     def add_bias(self, outputs):
#         output_rank = outputs.shape.rank
#         if self.rank == 1 and self._channels_first:
#             # nn.bias_add does not accept a 1D input tensor.
#             bias = array_ops.reshape(self.bias, (1, self.filters, 1))
#             outputs += bias
#         else:
#             # Handle multiple batch dimensions.
#             if output_rank is not None and output_rank > 2 + self.rank:
#                 def _apply_fn(o):
#                     return nn.bias_add(o, self.bias, data_format=self._tf_data_format)
#
#                 outputs = conv_utils.squeeze_batch_dims(
#                     outputs, _apply_fn, inner_rank=self.rank + 1)
#             else:
#                 outputs = nn.bias_add(
#                     outputs, self.bias, data_format=self._tf_data_format)
#         return outputs
#
#     def reconstruction(self):
#         return
#
#     def call(self, inputs, label=0, training=None):
#         patches = tf.image.extract_patches(
#             inputs,
#             [1, self.kernel_size, self.kernel_size, 1],
#             [1, self.strides, self.strides, 1],
#             [1, self.dilation_rate, self.dilation_rate, 1],
#             padding='VALID',
#         ) # B, n_patch, n_patch, patch_size(3*3)
#         distance_factor = self.distance_factor(self.kernel_size)
#         distance_factor = tf.reshape(distance_factor, [1, 1, 1, -1]) # [k, k] > [1, 1, 1, patch_size]
#         # patch encoding
#         patches *= distance_factor
#
#         outputs = self._convolution_op(inputs, self.kernel)
#
#         if self.use_bias:
#             outputs = self.add_bias(outputs)
#
#
#         if self.activation is not None:
#             return self.activation(outputs)
#         return outputs
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "weight": self.weight,
#         })
#         return config

########################################################################################################################
class channel_attention_base(Layer):
    def __init__(self, squeeze_rate=0.6):
        super(channel_attention_base, self).__init__()
        self.squeeze_rate = squeeze_rate
        self.BN1 = BatchNormalization()
        self.BN2 = BatchNormalization()

    def build(self, input_shape):
        self.n_channel = input_shape[-1]
        rank = len(input_shape)
        self.GAP = GlobalAveragePooling2D(keepdims=True) if rank == 4 \
            else GlobalAveragePooling3D(keepdims=True)
        self.squeezed_dense = Dense(self.squeeze_rate * self.n_channel)
        self.released_dense = Dense(self.n_channel)

    def get_config(self):
        config = super().get_config()
        config.update({
            'squeeze_rate': self.squeeze_rate,
        })
        return config


class SE(channel_attention_base):
    """
    Squeeze-and-Excitation Networks
        https://arxiv.org/abs/1709.01507
    """
    def __init__(self, squeeze_rate=0.6):
        super(SE, self).__init__(squeeze_rate)

    def call(self, inputs, training=None, **kargs):
        x = self.GAP(inputs)
        x = self.squeezed_dense(x)
        x = self.BN1(x)
        x = tf.nn.relu(x)

        x = self.released_dense(x)
        x = self.BN2(x)
        x = tf.math.sigmoid(x)
        return inputs * x

class SB(channel_attention_base):
    """
    Shift-and-Balance Attention
        https://arxiv.org/abs/2103.13080
    """
    def __init__(self, squeeze_rate=0.6):
        super(SB, self).__init__(squeeze_rate)

    def call(self, inputs, training=None, **kargs):
        x = self.GAP(inputs)
        x = self.squeezed_dense(x)
        x = self.BN1(x)
        x = tf.nn.relu(x)

        x = self.released_dense(x)
        x = self.BN2(x)
        x = tf.math.tanh(x)
        return inputs + x


########################################################################################################################
########################################################################################################################
# class Conv(Layer):
#     def __init__(self,
#                  rank,
#                  filters,
#                  kernel_size,
#                  strides=1,
#                  padding='valid',
#                  data_format=None,
#                  dilation_rate=1,
#                  groups=1,
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  trainable=True,
#                  name=None,
#                  conv_op=None,
#                  **kwargs):
#         super(Conv, self).__init__(trainable=trainable, name=name,
#                                    activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
#         self.rank = rank
#         if isinstance(filters, float):
#             filters = int(filters)
#         if filters is not None and filters < 0:
#             raise ValueError(f'Received a negative value for `filters`.'
#                              f'Was expecting a positive value, got {filters}.')
#         self.filters = filters
#         self.groups = groups or 1
#         self.kernel_size = conv_utils.normalize_tuple(
#             kernel_size, rank, 'kernel_size')
#         self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
#         self.padding = conv_utils.normalize_padding(padding)
#         self.data_format = conv_utils.normalize_data_format(data_format)
#         self.dilation_rate = conv_utils.normalize_tuple(
#             dilation_rate, rank, 'dilation_rate')
#
#         self.activation = activations.get(activation)
#         self.use_bias = use_bias
#
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)
#         self.input_spec = InputSpec(min_ndim=self.rank + 2)
#
#         self._validate_init()
#         self._is_causal = self.padding == 'causal'
#         self._channels_first = self.data_format == 'channels_first'
#         self._tf_data_format = conv_utils.convert_data_format(
#             self.data_format, self.rank + 2)
#
#     def _validate_init(self):
#         if self.filters is not None and self.filters % self.groups != 0:
#             raise ValueError(
#                 'The number of filters must be evenly divisible by the number of '
#                 'groups. Received: groups={}, filters={}'.format(
#                     self.groups, self.filters))
#
#         if not all(self.kernel_size):
#             raise ValueError('The argument `kernel_size` cannot contain 0(s). '
#                              'Received: %s' % (self.kernel_size,))
#
#         if not all(self.strides):
#             raise ValueError('The argument `strides` cannot contains 0(s). '
#                              'Received: %s' % (self.strides,))
#
#         if (self.padding == 'causal' and not isinstance(self,
#                                                         (Conv1D, SeparableConv1D))):
#             raise ValueError('Causal padding is only supported for `Conv1D`'
#                              'and `SeparableConv1D`.')
#
#     def build(self, input_shape):
#         input_shape = tensor_shape.TensorShape(input_shape)
#         input_channel = self._get_input_channel(input_shape)
#         if input_channel % self.groups != 0:
#             raise ValueError(
#                 'The number of input channels must be evenly divisible by the number '
#                 'of groups. Received groups={}, but the input has {} channels '
#                 '(full input shape is {}).'.format(self.groups, input_channel,
#                                                    input_shape))
#         kernel_shape = self.kernel_size + (input_channel // self.groups,
#                                            self.filters)
#
#         self.kernel = self.add_weight(
#             name='kernel',
#             shape=kernel_shape,
#             initializer=self.kernel_initializer,
#             regularizer=self.kernel_regularizer,
#             constraint=self.kernel_constraint,
#             trainable=True,
#             dtype=self.dtype)
#         if self.use_bias:
#             self.bias = self.add_weight(
#                 name='bias',
#                 shape=(self.filters,),
#                 initializer=self.bias_initializer,
#                 regularizer=self.bias_regularizer,
#                 constraint=self.bias_constraint,
#                 trainable=True,
#                 dtype=self.dtype)
#         else:
#             self.bias = None
#         channel_axis = self._get_channel_axis()
#         self.input_spec = InputSpec(min_ndim=self.rank + 2,
#                                     axes={channel_axis: input_channel})
#
#         # Convert Keras formats to TF native formats.
#         if self.padding == 'causal':
#             tf_padding = 'VALID'  # Causal padding handled in `call`.
#         elif isinstance(self.padding, str):
#             tf_padding = self.padding.upper()
#         else:
#             tf_padding = self.padding
#         tf_dilations = list(self.dilation_rate)
#         tf_strides = list(self.strides)
#
#         tf_op_name = self.__class__.__name__
#         if tf_op_name == 'Conv1D':
#             tf_op_name = 'conv1d'  # Backwards compat.
#
#         self._convolution_op = functools.partial(
#             nn_ops.convolution_v2,
#             strides=tf_strides,
#             padding=tf_padding,
#             dilations=tf_dilations,
#             data_format=self._tf_data_format,
#             name=tf_op_name)
#         self.built = True
#
#     def call(self, inputs):
#         input_shape = inputs.shape
#
#         if self._is_causal:  # Apply causal padding to inputs for Conv1D.
#             inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))
#
#         outputs = self._convolution_op(inputs, self.kernel)
#
#         if self.use_bias:
#             output_rank = outputs.shape.rank
#             if self.rank == 1 and self._channels_first:
#                 # nn.bias_add does not accept a 1D input tensor.
#                 bias = array_ops.reshape(self.bias, (1, self.filters, 1))
#                 outputs += bias
#             else:
#                 # Handle multiple batch dimensions.
#                 if output_rank is not None and output_rank > 2 + self.rank:
#
#                     def _apply_fn(o):
#                         return nn.bias_add(o, self.bias, data_format=self._tf_data_format)
#
#                     outputs = conv_utils.squeeze_batch_dims(
#                         outputs, _apply_fn, inner_rank=self.rank + 1)
#                 else:
#                     outputs = nn.bias_add(
#                         outputs, self.bias, data_format=self._tf_data_format)
#
#         if not context.executing_eagerly():
#             # Infer the static output shape:
#             out_shape = self.compute_output_shape(input_shape)
#             outputs.set_shape(out_shape)
#
#         if self.activation is not None:
#             return self.activation(outputs)
#         return outputs
#
#     def _spatial_output_shape(self, spatial_input_shape):
#         return [
#             conv_utils.conv_output_length(
#                 length,
#                 self.kernel_size[i],
#                 padding=self.padding,
#                 stride=self.strides[i],
#                 dilation=self.dilation_rate[i])
#             for i, length in enumerate(spatial_input_shape)
#         ]
#
#     def compute_output_shape(self, input_shape):
#         input_shape = tensor_shape.TensorShape(input_shape).as_list()
#         batch_rank = len(input_shape) - self.rank - 1
#         if self.data_format == 'channels_last':
#             return tensor_shape.TensorShape(
#                 input_shape[:batch_rank]
#                 + self._spatial_output_shape(input_shape[batch_rank:-1])
#                 + [self.filters])
#         else:
#             return tensor_shape.TensorShape(
#                 input_shape[:batch_rank] + [self.filters] +
#                 self._spatial_output_shape(input_shape[batch_rank + 1:]))
#
#     def _recreate_conv_op(self, inputs):  # pylint: disable=unused-argument
#         return False
#
#     def get_config(self):
#         config = {
#             'filters':
#                 self.filters,
#             'kernel_size':
#                 self.kernel_size,
#             'strides':
#                 self.strides,
#             'padding':
#                 self.padding,
#             'data_format':
#                 self.data_format,
#             'dilation_rate':
#                 self.dilation_rate,
#             'groups':
#                 self.groups,
#             'activation':
#                 activations.serialize(self.activation),
#             'use_bias':
#                 self.use_bias,
#             'kernel_initializer':
#                 initializers.serialize(self.kernel_initializer),
#             'bias_initializer':
#                 initializers.serialize(self.bias_initializer),
#             'kernel_regularizer':
#                 regularizers.serialize(self.kernel_regularizer),
#             'bias_regularizer':
#                 regularizers.serialize(self.bias_regularizer),
#             'activity_regularizer':
#                 regularizers.serialize(self.activity_regularizer),
#             'kernel_constraint':
#                 constraints.serialize(self.kernel_constraint),
#             'bias_constraint':
#                 constraints.serialize(self.bias_constraint)
#         }
#         base_config = super(Conv, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#     def _compute_causal_padding(self, inputs):
#         """Calculates padding for 'causal' option for 1-d conv layers."""
#         left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
#         if getattr(inputs.shape, 'ndims', None) is None:
#             batch_rank = 1
#         else:
#             batch_rank = len(inputs.shape) - 2
#         if self.data_format == 'channels_last':
#             causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
#         else:
#             causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
#         return causal_padding
#
#     def _get_channel_axis(self):
#         if self.data_format == 'channels_first':
#             return -1 - self.rank
#         else:
#             return -1
#
#     def _get_input_channel(self, input_shape):
#         channel_axis = self._get_channel_axis()
#         if input_shape.dims[channel_axis].value is None:
#             raise ValueError('The channel dimension of the inputs '
#                              'should be defined. Found `None`.')
#         return int(input_shape[channel_axis])
#
#     def _get_padding_op(self):
#         if self.padding == 'causal':
#             op_padding = 'valid'
#         else:
#             op_padding = self.padding
#         if not isinstance(op_padding, (list, tuple)):
#             op_padding = op_padding.upper()
#         return op_padding
