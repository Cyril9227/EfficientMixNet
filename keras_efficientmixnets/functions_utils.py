import glob
import logging
import os
from datetime import date
from itertools import islice
from math import ceil

import numpy as np
from keras.layers import Activation, BatchNormalization
from keras.utils import Sequence
from keras_efficientmixnets.custom_activations import Mish, Swish
from keras_efficientmixnets.custom_batchnorm import (AttentiveNormalization,
                                                     BatchAttNorm)


# currently supported type of activations and batchnorm
ACTIVATIONS = ["relu", "softmax", "sigmoid", "softplus", "mish", "swish"]
BATCH_NORM  = ["bn", "iebn", "an"]


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def round_filters(filters, width_coefficient, depth_divisor, min_depth):
    """Round number of filters based on depth multiplier."""
    multiplier = float(width_coefficient)
    divisor = int(depth_divisor)
    min_depth = min_depth

    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
def round_repeats(repeats, depth_coefficient):
    """Round number of filters based on depth multiplier."""
    multiplier = depth_coefficient

    if not multiplier:
        return repeats

    return int(ceil(multiplier * repeats))


def BatchNorm(typeBN="bn", batch_norm_momentum=0.99, batch_norm_epsilon=1e-3, batch_norm_axis=-1, k=5):
    """Wrapper to handle different types of batch normalization"""

    out_bn = BatchNormalization(axis=batch_norm_axis, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon)

    if typeBN is not None:
        if typeBN.lower() in ["attentive", "attention", "an"]:
            if isinstance(k, int):
                out_bn = AttentiveNormalization(n_mixture=k, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon, axis=batch_norm_axis)
            else:
                raise ValueError("k should be an integer but got : {}, {} instead".format(k, type(k)))

        elif typeBN.lower() in ["iebn", "instance"]:
            out_bn = BatchAttNorm(momentum=batch_norm_momentum, epsilon=batch_norm_epsilon, axis=batch_norm_axis)
            
    return out_bn




def activation(name='swish'):
    if isinstance(name, str):
        name = name.lower()
        if name in ACTIVATIONS:
            if name == 'mish':
                return Mish()
            elif name == 'swish':
                return Swish()
            else:
                return Activation(name)
        else:
            raise ValueError('Could not interpret {} as activation function, available activation functions are : {}'.format(name, ACTIVATIONS))
    else:
        raise ValueError('Please provide a string identifier, got {} instead'.format(name))



def get_activation_name(name='swish'):
    """Utility function to return activation name
    
    Arguments:
        name {str} -- input activation name as specified by the user (default='swish' as in the original paper).
    
    Raises:
        ValueError: if input activation name is not in the available activation list
        ValueError: if input activation name is not a string
    
    Returns:
        [str] -- An activation name which is in the available activation list 
    """
    if isinstance(name, str):
        name = name.lower()
        if name in ACTIVATIONS:
               return name
        else:
            raise ValueError('Could not interpret {} as activation function, available activation functions are : {}'.format(name, ACTIVATIONS))
    else:
        raise ValueError('Please provide a string identifier, got {} instead'.format(name))


def get_batchnorm_name(name='bn'):
    """Utility function to return batchnorm name
    
    Arguments:
        name {str} -- input batchnorm name as specified by the user (default='bn' as in the original paper). 
    
    Raises:
        ValueError: if input BN name is not in the available BN list
        ValueError: if input BN name is not a string
    
    Returns:
        [str] -- A BN name which is in the available BN list 
    """

    if isinstance(name, str):
        name = name.lower()
        if name in BATCH_NORM:
               return name
        else:
            raise ValueError('Could not interpret {} as batch norm, available batch norm are : {}'.format(name, BATCH_NORM))
    else:
        raise ValueError('Please provide a string identifier, got {} instead'.format(name))


class DataGenerator(Sequence):
    ### REM: les docstrings sont en général encodré de 3 guillemets.
    ###   Ca marche sans, mais c'est une convention, et les conventions
    ###   améliore la lisibilité du code pour tous
    'Generates data for Keras'
    def __init__(self, dic_paths_labels, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list(dic_paths_labels.items()) # [(path, label), ...]
        #### REM : Attention : l'ordre ici a t il de l'importance ?
        ####    Car la liste est ordonnée, le dictionnaire non (les clés peuvent 
        ####    apparaitrent dans des ordres random quand tu itères dessus ... )
        self.shuffle = shuffle
        self.datagen = self.__data_generation
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        return self.datagen(list_IDs_temp)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, 299, 299, 5)
        # Initialization

        #### REM : Cette méthode ressemble comme 2 gouttes d'eau
        ###     à load_data_set_from_pileups  => factoriser !
        X = np.empty((self.batch_size, 299, 299, 5))
        y = np.empty((self.batch_size, 2))

        # Generate data
        for i, item in enumerate(list_IDs_temp):
            X[i, ] = read_pgm_ref(item[0])
            # Store class
            y[i] = item[1]
        return X, y



def setup_logger(name, log_fname, folder_name, level=logging.INFO):
    """Function setup as many loggers as you want"""

    if not os.path.exists(folder_name):
        ### REM : utiliser l'option exist_ok = True
        os.makedirs(folder_name)

    log_file = folder_name + "/" + log_fname
    ### REM : utiliser os.path.join
    handler = logging.FileHandler(log_file)        
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
