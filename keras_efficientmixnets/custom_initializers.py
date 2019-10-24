import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.utils import get_custom_objects

__all__ = ['EfficientNetConvInitializer',
           'EfficientNetDenseInitializer',
           'ANInitializer']



# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
class EfficientNetConvInitializer(initializers.Initializer):
    """Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas base_path we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.

    # Arguments:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    # Returns:
      an initialization for the variable
    """
    def __init__(self):
        super(EfficientNetConvInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()

        kernel_height, kernel_width, _, out_filters = shape
        fan_out = int(kernel_height * kernel_width * out_filters)
        return K.random_normal(
            shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
class EfficientNetDenseInitializer(initializers.Initializer):
    """Initialization for dense kernels.
        This initialization is equal to
          tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                          distribution='uniform').
        It is written out explicitly base_path for clarity.

        # Arguments:
          shape: shape of variable
          dtype: dtype of variable
          partition_info: unused

        # Returns:
          an initialization for the variable
    """
    def __init__(self):
        super(EfficientNetDenseInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()

        init_range = 1.0 / np.sqrt(shape[1])
        return K.random_uniform(shape, -init_range, init_range, dtype=dtype)



class ANInitializer(initializers.Initializer):
    """Initialization for gamma and beta weights according to BigGan paper 
    (A. Brock, J. Donahue, and K. Simonyan. Large scale gan
    training for high fidelity natural image synthesis. arXiv
    preprint arXiv:1809.11096, 2018.)
    
        This initialization is equal to :  scale * N(0, 1) + bias
         
        # Arguments:
          scale: rescaling factor
          bias: bias factor
          shape: shape of variable
          dtype: dtype of variable
          seed: random seed for reprocudibility
        # Returns:
          an initialization for the variable
          
    """
    def __init__(self, scale=0.1, bias=0., seed=1997):
        super(ANInitializer, self).__init__()
        self.scale = scale
        self.bias = bias
        self.seed = seed

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()
        return self.scale * K.random_normal(shape=shape, mean=0.0, stddev=1., seed=self.seed) + self.bias



get_custom_objects().update({
    'ANInitializer': ANInitializer,
    'EfficientNetDenseInitializer' : EfficientNetDenseInitializer,
    'EfficientNetConvInitializer' : EfficientNetConvInitializer

})
