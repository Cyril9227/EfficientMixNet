import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model



from tensorflow.keras.utils import get_custom_objects

from keras_efficientmixnets.custom_activations import Swish
from keras_efficientmixnets.custom_initializers import \
    EfficientNetConvInitializer
from keras_efficientmixnets.functions_utils import BatchNorm, activation, conv_output_length

__all__ = ['DropConnect',
           'GroupDepthwiseConvolution',
           'MDConv',
           'SEBlock',
           'MBConvBlock']



# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def SEBlock(input_filters, se_ratio, expand_ratio, data_format=None, name="swish"):
    if data_format is None:
        data_format = K.image_data_format()

    num_reduced_filters = max(1, int(input_filters * se_ratio))   
    filters = input_filters * expand_ratio

    if data_format == 'channels_first':
        spatial_dims = [2, 3]
    else:
        spatial_dims = [1, 2]

    def block(inputs):
        x = inputs
        # Squeeze
        x = layers.Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = layers.Conv2D(
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=True)(x)
        x = activation(name)(x)

        # Excite
        x = layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=True)(x)
        x = activation('sigmoid')(x)
        out = layers.Multiply()([x, inputs])
        return out

    return block

# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def MBConvBlock(input_filters, output_filters,
                kernel_size, strides,
                expand_ratio, se_ratio,
                id_skip, drop_connect_rate,
                data_format=None, name="swish", typeBN="bn", n_mixture=None, batch_norm_momentum=0.99, batch_norm_epsilon=0.001):

    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = int(input_filters * expand_ratio)

    def block(inputs):
        if expand_ratio != 1:
            x = layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=EfficientNetConvInitializer(),
                padding='same',
                use_bias=False)(inputs)
            x = BatchNorm(
                typeBN=typeBN, 
                k=n_mixture, 
                batch_norm_momentum=batch_norm_momentum, 
                batch_norm_epsilon=batch_norm_epsilon, 
                batch_norm_axis=channel_axis)(x) 
            x = activation(name)(x)           
        else:
            x = inputs

        if isinstance(kernel_size, int):
            x = layers.DepthwiseConv2D(
                [kernel_size, kernel_size],
                strides=strides,
                depthwise_initializer=EfficientNetConvInitializer(),
                padding='same',
                use_bias=False)(x)

        else:
            x = MDConv(
                kernel_size,
                strides=strides,
                depthwise_initializer=EfficientNetConvInitializer(),
                padding='same',
                use_bias=False)(x)

            x = BatchNorm(
                typeBN=typeBN, 
                k=n_mixture, 
                batch_norm_momentum=batch_norm_momentum, 
                batch_norm_epsilon=batch_norm_epsilon, 
                batch_norm_axis=channel_axis)(x)
        x = activation(name)(x)
        
        if has_se:
            x = SEBlock(input_filters, se_ratio, expand_ratio,
                        data_format, name)(x)

        # output phase
        x = layers.Conv2D(
            output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=EfficientNetConvInitializer(),
            padding='same',
            use_bias=False)(x)

        x = BatchNorm(
            typeBN=typeBN, 
            k=n_mixture, 
            batch_norm_momentum=batch_norm_momentum, 
            batch_norm_epsilon=batch_norm_epsilon, 
            batch_norm_axis=channel_axis)(x)

        if id_skip:
            if all(s == 1 for s in strides) and (
                    input_filters == output_filters):
                # only apply drop_connect if skip presents.
                if drop_connect_rate:
                    x = DropConnect(drop_connect_rate)(x)
                x = layers.Add()([x, inputs])
        return x

    return block




# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
class DropConnect(layers.Layer):
    def __init__(self, drop_connect_rate=0., **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.drop_connect_rate = float(drop_connect_rate)

    def call(self, inputs, training=None):

        def drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate
            # Compute drop_connect tensor
            batch_size = K.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = (inputs / keep_prob) * binary_tensor
            return output

        return K.in_train_phase(drop_connect, inputs, training=training)

    def get_config(self):
        config = {'drop_connect_rate': self.drop_connect_rate}
        base_config = super(DropConnect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class GroupDepthwiseConvolution(Model):
  """Model subclassing for proper serialization
        to do : dilated conv, add more options such as depthwise multiplier etc
        warning : model.save() is not available when using this, please use load_weights() instead.
    """

  def __init__(self, filters, kernels, groups, data_format=None, conv_kwargs=None, **kwargs):

      super(GroupDepthwiseConvolution, self).__init__(**kwargs)

      default_conv_kwargs = {
            'strides': (1, 1),
            'padding': 'same',
            'use_bias': False}

      if conv_kwargs is None:
          conv_kwargs = default_conv_kwargs  
      else :
          conv_kwargs = {**default_conv_kwargs, **conv_kwargs} 
          
      if isinstance(kernels, int):
          kernels = [kernels]
  
      self.filters = filters
      self.kernels = kernels
      self.groups = groups
      self.strides = conv_kwargs['strides']
      self.padding = conv_kwargs['padding']
      self.use_bias = conv_kwargs['use_bias']
      self.conv_kwargs = conv_kwargs
      self._layers = [layers.DepthwiseConv2D(kernels[i],
                                              strides=self.strides,
                                              padding=self.padding,
                                              use_bias=self.use_bias,
                                              kernel_initializer=EfficientNetConvInitializer())
                      for i in range(groups)]
      
      if data_format is None:
          data_format = K.image_data_format() # returns 'channels_first' or 'channels_last'

      if data_format == 'channels_first':
          self._channel_axis = 1
      else:
          self._channel_axis = -1
          
  def _split_channels(self, total_filters, num_groups):
      split = [total_filters // num_groups for _ in range(num_groups)]
      split[0] += total_filters - sum(split)
      return split        
          
  def call(self, inputs, **kwargs):
      if len(self._layers) == 1:
          return self._layers[0](inputs)

      filters = K.int_shape(inputs)[self._channel_axis]
      splits = self._split_channels(filters, self.groups)
      x_splits = tf.split(inputs, splits, self._channel_axis)
      x_outputs = [c(x) for x, c in zip(x_splits, self._layers)]
      x = layers.concatenate(x_outputs, axis=self._channel_axis)
      return x

  def compute_output_shape(self, input_shape):
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
          new_dim = conv_output_length(
              space[i],
              filter_size=1,
              padding=self.padding,
              stride=self.strides[i])
          new_space.append(new_dim)
      return (input_shape[0],) + tuple(new_space) + (self.filters,)


  def get_config(self):
      config = {
          'filters': self.filters,
          'kernels': self.kernels,
          'groups': self.groups,
          'strides': self.conv_kwargs['strides'],
          'padding': self.conv_kwargs['padding'],
          'conv_kwargs': self.conv_kwargs}
      
      base_config = super(GroupDepthwiseConvolution, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))



class MDConv(object):
  """MDConv with mixed depthwise convolutional kernels.
  MDConv is an improved depthwise convolution that mixes multiple kernels (e.g.
  3x3, 5x5, etc). Right now, we use an naive implementation that split channels
  into multiple groups and perform different kernels for each group.
  See Mixnet paper for more details.
  """

  def __init__(self, kernel_size, strides, data_format=None, **kwargs):
      """Initialize the layer.
      Most of args are the same as tf.keras.layers.DepthwiseConv2D except it has
      an extra parameter "dilated" to indicate whether to use dilated conv to
      simulate large kernel_size size. If dilated=True, then dilation_rate is ignored.
      Args:
        kernel_size: An integer or a list. If it is a single integer, then it is
          same as the original keras.layers.DepthwiseConv2D. If it is a list,
          then we split the channels and perform different kernel_size for each group.
        strides: An integer or tuple/list of 2 integers, specifying the strides of
          the convolution along the height and width.
        data_format: String or None. indicate the channel axis with 'channels_first' or 'channels_last'
           default value = 'channels_last'
        **kwargs: other parameters passed to the original depthwise_conv layer.
      """
      if data_format is None:
          data_format = K.image_data_format() # returns 'channels_first' or 'channels_last'

      if data_format == 'channels_first':
          self._channel_axis = 1
      else:
          self._channel_axis = -1
          
          
      if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

      self.kernels = kernel_size

      self._conv_kwargs = {
          'strides': strides,
          'kernel_initializer': kwargs.get('kernel_initializer', EfficientNetConvInitializer()),
          'padding': 'same',
          'use_bias': kwargs.get('use_bias', False),
      }

  def __call__(self, inputs):
      filters = K.int_shape(inputs)[self._channel_axis]
      grouped_op = GroupDepthwiseConvolution(filters, self.kernels, groups=len(self.kernels),
                                    conv_kwargs=self._conv_kwargs)
      return grouped_op(inputs)


get_custom_objects().update({
    'DropConnect': DropConnect,
    'MDConv' : MDConv,
    'GroupDepthwiseConvolution' : GroupDepthwiseConvolution,
    'SEBlock' : SEBlock,
    'MBConvBlock' : MBConvBlock
})
