# -*- coding: utf-8 -*-

from keras_efficientmixnets.custom_activations import Swish
from keras_efficientmixnets.custom_initializers import EfficientNetDenseInitializer
from keras_efficientmixnets.custom_optimizers import get, Lookahead
from keras_efficientmixnets.efficientmixnet import EfficientNetBuilder

import argparse
import json
import os
from pathlib import Path

import keras.backend as K
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Input
from keras.metrics import categorical_accuracy
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split


from keras_efficientmixnets.functions_utils import DataGenerator, load_data_set_from_pileups, setup_logger
from datetime import date

K.clear_session()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

"""
Created on Wed Jul 24 15:48:03 2019
@author: cec1
"""





def create_network(input_shape, keras_model, mixed=True, typeBN=None):
    '''Conv Network architecture, based on a keras model such as EfficientNets
    '''
    inp = Input(shape=input_shape)
    base_model = keras_model(input_tensor=inp, weights=None, include_top=False, input_shape=input_shape, pooling="avg", mixed=mixed)
    x = base_model.output
    x = Dense(1024, use_bias=True, kernel_initializer=EfficientNetDenseInitializer())(x)
    x = Swish()(x)
    x = Dropout(0.5)(x)
    x = Dense(200, use_bias=True, kernel_initializer=EfficientNetDenseInitializer())(x)
    x = Swish()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(2, activation="softmax", kernel_initializer=EfficientNetDenseInitializer())(x)
    return Model(inp, predictions)



def train(args):

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    
    with open(args.path_partition, 'r') as f:
        logging.info("Loading partition of the data...")
        partition = json.load(f)
        logging.info("Partition loaded...")

    dics = [partition["paths_labels_" + chrom + "_down"] if args.is_downsampled else partition["paths_labels_" + chrom] for chrom in args.chromosom]
    paths_labels = {}
    for d in dics:
        paths_labels.update(d)
    if args.is_downsampled:
        locs_train, locs_valid = train_test_split(list(paths_labels.keys()), test_size=0.2, random_state=1997)
    else:
        locs_train, locs_valid = train_test_split(list(paths_labels.keys()), test_size=0.2, random_state=1997, 
                                                    stratify=np.argmax(list(paths_labels.values()), axis=1))

    paths_labels_train = {loc : paths_labels[loc] for loc in locs_train}
    paths_labels_valid = {loc : paths_labels[loc] for loc in locs_valid}

    logging.info("Training on {} images".format(len(locs_train)))
    logging.info("Validation on {} images".format(len(locs_valid)))

    
    if args.architecture in ['custom', 'Custom']:
        if Path(str(args.path_model)).exists():
            logging.info("Loading custom model from: {} ".format(args.path_model))
            custom_objects = args.custom_objects or {}
            model = load_model(str(args.path_model), custom_objects)
            logging.info("Custom model loaded")
        else:
            raise ValueError("Path to custom model architecture doesn't exists !")
    else:
        logging.info("Instantiating {} architecture...".format(args.architecture))
        module = __import__('keras_efficientmixnets.efficientmixnet')
        builder = EfficientNetBuilder(input_shape=(299, 299, 5),
                                   include_top=True,
                                   weights=None,
                                   input_tensor=None,
                                   pooling='avg',
                                   classes=2,
                                   drop_connect_rate=0.,
                                   data_format=None,
                                   mixed=False,
                                   activation="swish",
                                   typeBN="bn",
                                   batch_norm_momentum=0.99,
                                   batch_norm_epsilon=0.001,
                                   n_mixture=None,
                                   depth_divisor=8,
                                   min_depth=None);
        #model = create_network((299, 299, 5), getattr(module, args.architecture), mixed=args.mixed) # calls EfficientNetBX from a string 
        
        model = builder.default_model("B2")
        if args.path_model is not None:
            if Path(str(args.path_model)).exists():
                model.load_weights(args.path_model)
                logging.info("Pretrained weights loaded")
            else:
                raise ValueError("Path to pretrained weights doesn't exists !")

    opt = get(args.optimizer)
    logging.info("Compiling model with : {} - {} optimizer".format(args.optimizer, opt))
    model.compile(opt, loss='categorical_crossentropy', metrics=[categorical_accuracy])
    logging.info("Model compiled")
    if args.lookahead:
        lookahead = Lookahead(k=5, alpha=0.5) # Initialize Lookahead
        lookahead.inject(model) # add into model
        logging.info("Using Lookahead")

    folder_name = "outputs/model_checkpoints_" + str(date.today())
    #### REM: folder_name est un paramètre. Peut-être qu'un jour tu auras envie que l'output
    ###    soit dans un dossier /tmp/outputs/  ou bien outputs2/
    ###    Il faut donc que le dossier de sortie, soit demandé en paramètres
    ###     et qu'il est la valeur par défaut "outputs".
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    ### REM: il y a un raccourci pour ca !
    ###   cf  https://docs.python.org/3/library/os.html#os.makedirs
    ###   avec l'option exist_ok
    
    model_check_point_name = folder_name + "/" + "model_" + args.architecture + ".h5"
    ### REM tu as utiliser Path ou os.path précédemment, c'est très bien
    ###   Il faut continuer à faire pareil car là le "/" que tu mets
    ###   selon si t'es sous windows ou linux, ce symbole de séparation des dossiers
    ###   ne sera pas le même. Les librairies python gère ca pour toi
    ###   Tu peux utiliser notamment os.path.join("dossier1", "sous-dossier", "fichier")
    ###   par exemple (tu peux donner un nombre d'arguments arbitraire à cette fonction)    
    
    log = setup_logger("training_config", "training_config.log", folder_name)
    log.info("weights obtained by calling training.py with args : {}".format(args))
    early_stopping = EarlyStopping(monitor="val_categorical_accuracy", patience=args.patience, restore_best_weights=True, mode="max", verbose=1)
    model_checkpoint = ModelCheckpoint(model_check_point_name, 
                                        monitor="val_categorical_accuracy", save_best_only=True, 
                                        save_weights_only=args.save_weights_only, mode='max') # store whole model (architecture + weights) or only weights
    my_callbacks = [model_checkpoint, early_stopping]
    
    if args.in_memory:
        logging.info("Training in memory...")
        
        X_train, y_train = load_data_set_from_pileups(paths_labels_train)
        X_valid, y_valid = load_data_set_from_pileups(paths_labels_valid)

        logging.info("...Using batch size of: {}, for {} epochs".format(args.batch_size, args.my_epochs))
        
        model.fit(X_train, y_train, epochs=args.my_epochs, batch_size=args.batch_size,
                            validation_data=(X_valid, y_valid),
                            callbacks=my_callbacks)
    else:
        logging.info("Training using a data generator...")

        batch_train = DataGenerator(paths_labels_train, batch_size=args.batch_size)
        batch_valid = DataGenerator(paths_labels_valid,  batch_size=args.batch_size)

        logging.info("...for {} epochs".format(args.my_epochs))

        model.fit_generator(batch_train, epochs=args.my_epochs, callbacks=my_callbacks, validation_data=batch_valid)




if __name__ == "__main__":
    # execute only if run as a script
    
    folder_name = "outputs/logs_" + str(date.today())
    logging = setup_logger("training_logger", "training.log", folder_name)

    parser = argparse.ArgumentParser(description='Training Script for Efficient Networks')
    parser.add_argument('-p', '--partition', dest='path_partition', required=True, type=str, help='path to partition containing list of index for training images')
 
    parser.add_argument('--downsample', dest='is_downsampled', action='store_true', help='whether to use less training images or not')
    parser.add_argument('--no_downsample', dest='is_downsampled', action='store_false', help='whether to use less training images or not')
   

    parser.add_argument('--memory', dest='in_memory', action='store_true', help='whether to load training / validation in memory or use a DataGenerator')
    parser.add_argument('--no_memory', dest='in_memory', action='store_false', help='whether to load training / validation in memory or use a DataGenerator')

    parser.add_argument('-chr', '--chromosom', nargs='+', help="Chromosom(s) for training")

    parser.add_argument('--architecture', default='EfficientNetB2', type=str, 
                        choices=['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'custom', 'Custom'], 
                        help='Neural network architecture, can be created from a file')

    parser.add_argument('--path_model', default=None, type=str, help='Path to custom Neural Network architecture or pretrained weights')
    parser.add_argument('-co', '--custom_objects', default={}, type=dict, help='dict of customs objects / layers in case of using model.load(), to-do : find a way to import the propers custom_objects such as swish etc')
    
    parser.add_argument('--mixed', dest='mixed', action='store_true', help='Whether to use mixed depthwise convolution when using default efficient nets architectures')
    parser.add_argument('--no_mixed', dest='mixed', action='store_false', help='Whether to use mixed depthwise convolution when using default efficient nets architectures')

    parser.add_argument('-tbn', '--typeBN', dest='typeBN', default="Default BN", type=str, help='type of BatchNorm to use, choices are : "attentive", "an", "iebn"')
    parser.add_argument('--n_mixture', dest='n_mixture', default=5, type=int, help='only relevant when using typeBN = "attentive", number of mixture to consider when learning the affine BN transformation')


    parser.add_argument('-opt', '--optimizer',  default='Adam', type=str, help='Choice of the optimizer')

    parser.add_argument('--lookahead', dest='lookahead', action='store_true', help='Whether to use look ahead for optimization or not')
    parser.add_argument('--no_lookahead', dest='lookahead', action='store_false', help='Whether to use look ahead for optimization or not')

    parser.add_argument('-pat', '--patience',  default=15, type=int, help='Patience for early stopping')
    
    parser.add_argument('--save_weights_only',  dest='save_weights_only', action='store_true', help='Save weights only option for ModelCheckpoint, if using model subclassing please set it to True')
    parser.add_argument('--no_save_weights_only',  dest='save_weights_only', action='store_false', help='Save weights only option for ModelCheckpoint, if using model subclassing please set it to True')
    

    parser.add_argument('-b', '--batchsize', dest='batch_size', default=64, type=int, help='batch size')
    parser.add_argument('-e', '--epoch', dest='my_epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--gpu', dest='gpu_id', default='1', type=str, help='ID of the GPU to use')

    parser.set_defaults(lookahead=True, save_weights_only=True, mixed=True, in_memory=True, is_downsampled=True)
    args = parser.parse_args()


    train(args)
