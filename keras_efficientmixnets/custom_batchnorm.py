import tensorflow.keras.backend as K
from tensorflow.keras import initializers, layers
from tensorflow.keras.utils import get_custom_objects

from keras_efficientmixnets.custom_initializers import ANInitializer

__all__ = ['AttentiveNormalization', 'BatchAttNorm']


class AttentiveNormalization(layers.BatchNormalization):
    def __init__(self, n_mixture=5, momentum=0.99, epsilon=0.1, axis=-1, **kwargs):
        super(AttentiveNormalization, self).__init__(momentum=momentum, epsilon=epsilon, axis=axis, center=False, scale=False, **kwargs)

        if self.axis == -1:
            self.data_format = 'channels_last'
        else:
            self.data_format = 'channel_first'
            
        self.n_mixture = n_mixture
        
    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(len(input_shape)))
            
        super(AttentiveNormalization, self).build(input_shape)
        
        dim = input_shape[self.axis]
        shape = (self.n_mixture, dim) # K x C 
        
        self.FC = layers.Dense(self.n_mixture, activation="sigmoid")
        self.FC.build(input_shape) # (N, C)
        
        self.GlobalAvgPooling = layers.GlobalAveragePooling2D(self.data_format)
        self.GlobalAvgPooling.build(input_shape)
        
        self._trainable_weights = self.FC.trainable_weights
        
        self.learnable_weights = self.add_weight(name='gamma2', 
                                      shape=shape,
                                      initializer=ANInitializer(scale=0.1, bias=1.),
                                      trainable=True)

        self.learnable_bias = self.add_weight(name='bias2', 
                                    shape=shape,
                                    initializer=ANInitializer(scale=0.1, bias=0.),
                                    trainable=True)
        

    def call(self, inputs):
        # input is a batch of shape : (N, H, W, C)
        avg = self.GlobalAvgPooling(inputs) # N x C 
        attention = self.FC(avg) # N x K 
        gamma_readjust = K.dot(attention, self.learnable_weights) # N x C
        beta_readjust  = K.dot(attention, self.learnable_bias)  # N x C
        
        out_BN = super(AttentiveNormalization, self).call(inputs) # rescale input, N x H x W x C

        # broadcast if needed
        if K.int_shape(inputs)[0] is None or K.int_shape(inputs)[0] > 1:
            gamma_readjust = gamma_readjust[:, None, None, :]
            beta_readjust  = beta_readjust[:, None, None, :]

        return gamma_readjust * out_BN + beta_readjust

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'n_mixture' : self.n_mixture
        }
        base_config = super(AttentiveNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class BatchAttNorm(layers.BatchNormalization):
    def __init__(self, momentum=0.99, epsilon=0.001, axis=-1, **kwargs):
        super(BatchAttNorm, self).__init__(momentum=momentum, epsilon=epsilon, axis=axis, center=False, scale=False, **kwargs)
        
        if self.axis == -1:
            self.data_format = 'channels_last'
        else:
            self.data_format = 'channel_first'
        
    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input_shape))
                
        super(BatchAttNorm, self).build(input_shape)   
        
        dim = input_shape[self.axis]
        shape = (dim, )
        
        self.GlobalAvgPooling = layers.GlobalAveragePooling2D(self.data_format)
        self.GlobalAvgPooling.build(input_shape)
    
        self.weight = self.add_weight(name='weight', 
                                      shape=shape,
                                      initializer=initializers.Constant(1),
                                      trainable=True)

        self.bias = self.add_weight(name='bias', 
                                    shape=shape,
                                    initializer=initializers.Constant(0),
                                    trainable=True)

        self.weight_readjust = self.add_weight(name='weight_readjust', 
                                               shape=shape,
                                               initializer=initializers.Constant(0),
                                               trainable=True)
        
        self.bias_readjust = self.add_weight(name='bias_readjust', 
                                             shape=shape,
                                             initializer=initializers.Constant(-1),
                                             trainable=True)
        

    def call(self, inputs):
        avg = self.GlobalAvgPooling(inputs) 
        attention = K.sigmoid(avg * self.weight_readjust + self.bias_readjust)

        bn_weights = self.weight * attention          
        
        out_bn = super(BatchAttNorm, self).call(inputs)
        
        if K.int_shape(inputs)[0] is None or K.int_shape(inputs)[0] > 1:
            bn_weights = bn_weights[:, None, None, :]
            self.bias  = self.bias[None, None, None, :]
 
        return out_bn * bn_weights + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


get_custom_objects().update({
    'BatchAttNorm': BatchAttNorm,
    'AttentiveNormalization' : AttentiveNormalization
})
