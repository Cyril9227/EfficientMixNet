# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for EfficientNet model.
[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

import datetime
import os
from math import ceil
from typing import List

import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.models import Model
from keras.utils import get_file, get_source_inputs
from keras_applications.imagenet_utils import _obtain_input_shape

from keras_efficientmixnets.functions_utils import BatchNorm, activation, round_filters, round_repeats

from keras_efficientmixnets.config import BlockArgs, get_block_list
from keras_efficientmixnets.custom_activations import Mish, Swish
from keras_efficientmixnets.custom_initializers import (
    ANInitializer, EfficientNetConvInitializer, EfficientNetDenseInitializer)
from keras_efficientmixnets.custom_blocs import DropConnect, MDConv, SEBlock, MBConvBlock
from keras_efficientmixnets.functions_utils import setup_logger


class EfficientNetBuilder:
    def __init__(self, input_shape=None,
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   pooling=None,
                   classes=1000,
                   drop_connect_rate=0.,
                   data_format=None,
                   mixed=False,
                   typeBN=None,
                   n_mixture=None,
                   batch_norm_momentum=0.99,
                   batch_norm_epsilon=1e-3,
                   activation="mish",
                   depth_divisor=8,
                   min_depth=None):

                   self.input_shape=input_shape
                   self.include_top=include_top
                   self.weights=weights
                   self.input_tensor=input_tensor
                   self.pooling=pooling
                   self.classes=classes
                   self.drop_connect_rate=drop_connect_rate
                   self.data_format=data_format
                   self.mixed=mixed
                   self.typeBN=typeBN
                   self.n_mixture=n_mixture
                   self.batch_norm_momentum=batch_norm_momentum
                   self.batch_norm_epsilon=batch_norm_epsilon
                   self.activation=activation
                   self.depth_divisor=depth_divisor
                   self.min_depth=min_depth
                   

                   self.params = {
                        "B0" : {'width_coefficient' : 1.0, 'depth_coefficient' : 1.0, 'default_size' : 224, 'dropout_rate' : 0.2},
                        "B1" : {'width_coefficient' : 1.0, 'depth_coefficient' : 1.1, 'default_size' : 240, 'dropout_rate' : 0.2},
                        "B2" : {'width_coefficient' : 1.1, 'depth_coefficient' : 1.2, 'default_size' : 260, 'dropout_rate' : 0.3},
                        "B3" : {'width_coefficient' : 1.2, 'depth_coefficient' : 1.4, 'default_size' : 300, 'dropout_rate' : 0.3},
                        "B4" : {'width_coefficient' : 1.4, 'depth_coefficient' : 1.8, 'default_size' : 380, 'dropout_rate' : 0.4},
                        "B5" : {'width_coefficient' : 1.6, 'depth_coefficient' : 2.2, 'default_size' : 456, 'dropout_rate' : 0.4},
                        "B6" : {'width_coefficient' : 1.8, 'depth_coefficient' : 2.6, 'default_size' : 528, 'dropout_rate' : 0.5},
                        "B7" : {'width_coefficient' : 2.0, 'depth_coefficient' : 3.1, 'default_size' : 600, 'dropout_rate' : 0.5}
                        }
    @staticmethod
    def make_model(input_shape,   
                 block_args_list: List[BlockArgs],  
                 width_coefficient: float,
                 depth_coefficient: float,
                 include_top=True,
                 weights=None,
                 input_tensor=None,
                 pooling=None,
                 classes=1000,
                 dropout_rate=0.,
                 drop_connect_rate=0.,
                 depth_divisor=8,
                 min_depth=None,
                 data_format=None,
                 default_size=224,
                 mixed=False,
                 name="swish",
                 typeBN="bn",
                 n_mixture=None,
                 batch_norm_momentum=0.99,
                 batch_norm_epsilon=1e-3,
                 **kwargs):

                 logs_path = os.path.join("outputs/logs_" + str(datetime.datetime.now()).split(".")[0][:-3].replace(":", "h"))
                 logging = setup_logger("model_logger", "model_instantiation.log", logs_path) 
                 
                 
                 if not (weights in {'imagenet', None} or os.path.exists(weights)):
                    raise ValueError('The `weights` argument should be either '
                                    '`None` (random initialization), `imagenet` '
                                    '(pre-training on ImageNet), '
                                    'or the path to the weights file to be loaded.')

                 if weights == 'imagenet' and include_top and classes != 1000:
                    raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                                    'as true, `classes` should be 1000')

                 if data_format is None:
                     data_format = K.image_data_format()

                 if data_format == 'channels_first':
                     channel_axis = 1
                 else:
                     channel_axis = -1

                 if block_args_list is None:
                     block_args_list = get_block_list(mixed)

                # count number of strides to compute min size
                 stride_count = 1
                 for block_args in block_args_list:
                     if block_args.strides is not None and block_args.strides[0] > 1:
                         stride_count += 1

                 min_size = int(2 ** stride_count)

                # Determine proper input shape and default size.
                 input_shape = _obtain_input_shape(input_shape,
                                                default_size=default_size,
                                                min_size=min_size,
                                                data_format=data_format,
                                                require_flatten=include_top,
                                                weights=weights)

                # Stem part
                 if input_tensor is None:
                    inputs = layers.Input(shape=input_shape)
                 else:
                     if not K.is_keras_tensor(input_tensor):
                         inputs = layers.Input(tensor=input_tensor, shape=input_shape)
                     else:
                         inputs = input_tensor

                 x = inputs
                 x = layers.Conv2D(
                    filters=round_filters(32, width_coefficient,
                                        depth_divisor, min_depth),
                    kernel_size=[3, 3],
                    strides=[2, 2],
                    kernel_initializer=EfficientNetConvInitializer(),
                    padding='same',
                    use_bias=False)(x)
                 x = BatchNorm(
                    typeBN=typeBN, 
                    k=n_mixture, 
                    batch_norm_momentum=batch_norm_momentum, 
                    batch_norm_epsilon=batch_norm_epsilon, 
                    batch_norm_axis=channel_axis)(x)    
                 x = activation(name)(x)

                 num_blocks = sum([block_args.num_repeat for block_args in block_args_list])
                 drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)

                # Blocks part
                 logging.info("Encoded bloc strings used for this model are :")
                 for block_idx, block_args in enumerate(block_args_list):
                     assert block_args.num_repeat > 0
                     logging.info(block_args.encode_block_string(block_args))

                    # Update block input and output filters based on depth multiplier.
                     block_args.input_filters  = round_filters(block_args.input_filters, width_coefficient, depth_divisor, min_depth)
                     block_args.output_filters = round_filters(block_args.output_filters, width_coefficient, depth_divisor, min_depth)
                     block_args.num_repeat = round_repeats(block_args.num_repeat, depth_coefficient)

                    # The first block needs to take care of stride and filter size increase.
                     x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                                    block_args.kernel_size, block_args.strides,
                                    block_args.expand_ratio, block_args.se_ratio,
                                    block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                                    data_format, block_args.activation, block_args.typeBN, block_args.n_mixture, batch_norm_momentum, batch_norm_epsilon)(x)

                     if block_args.num_repeat > 1:
                         block_args.input_filters = block_args.output_filters
                         block_args.strides = [1, 1]

                     for _ in range(block_args.num_repeat - 1):
                         x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                                        block_args.kernel_size, block_args.strides,
                                        block_args.expand_ratio, block_args.se_ratio,
                                        block_args.identity_skip, drop_connect_rate_per_block * block_idx, 
                                        data_format, block_args.activation, block_args.typeBN, block_args.n_mixture, batch_norm_momentum, batch_norm_epsilon)(x)
                # Head part
                 x = layers.Conv2D(
                    filters=round_filters(1280, width_coefficient, depth_coefficient, min_depth),
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
                 x = activation(name)(x)
                
                 if pooling == 'avg':
                     x = layers.GlobalAveragePooling2D()(x)
                 elif pooling == 'max':
                     x = layers.GlobalMaxPooling2D()(x)
                 elif pooling is None:
                     x = layers.Flatten()(x)
                 else:
                     raise ValueError("pooling type not understood, expected 'avg', 'max' or None but got {} instead".format(pooling))

                 if include_top:
                     if dropout_rate > 0:
                         x = layers.Dropout(dropout_rate)(x)
                     x = layers.Dense(classes, kernel_initializer=EfficientNetDenseInitializer())(x)
                     x = activation('softmax')(x)

                 outputs = x
                # Ensure that the model takes into account
                # any potential predecessors of `input_tensor`.
                 if input_tensor is not None:
                     inputs = get_source_inputs(input_tensor)

                 model = Model(inputs, outputs)

                 if weights is not None:
                     model.load_weights(weights)

                 return model


    def default_model(self, BX):
        return self.make_model(self.input_shape,
                        get_block_list(self.mixed),
                        include_top=self.include_top,
                        weights=self.weights,
                        input_tensor=self.input_tensor,
                        pooling=self.pooling,
                        classes=self.classes,
                        drop_connect_rate=self.drop_connect_rate,
                        name=self.activation,
                        data_format=self.data_format,
                        mixed=self.mixed,
                        typeBN=self.typeBN,
                        n_mixture=self.n_mixture,
                        batch_norm_momentum=self.batch_norm_momentum,
                        batch_norm_epsilon=self.batch_norm_epsilon,
                        activation=self.activation,
                        depth_divisor=self.depth_divisor,
                        min_depth=self.min_depth,
                        **self.params[BX])
        
    def custom_model(self, block_list, width_coefficient=1.1, depth_coefficient=1.2, default_size=260, dropout_rate=0.3):
        return self.make_model(self.input_shape,
                block_list,
                include_top=self.include_top,
                weights=self.weights,
                input_tensor=self.input_tensor,
                pooling=self.pooling,
                classes=self.classes,
                drop_connect_rate=self.drop_connect_rate,
                name=self.activation,
                data_format=self.data_format,
                mixed=self.mixed,
                typeBN=self.typeBN,
                n_mixture=self.n_mixture,
                batch_norm_momentum=self.batch_norm_momentum,
                batch_norm_epsilon=self.batch_norm_epsilon,
                activation=self.activation,
                depth_divisor=self.depth_divisor,
                min_depth=self.min_depth,
                width_coefficient=width_coefficient, 
                depth_coefficient=depth_coefficient, 
                default_size=default_size, 
                dropout_rate=dropout_rate)
