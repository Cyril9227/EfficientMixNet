#!/usr/bin/env python
import argparse
import json
import os
from datetime import date
from pathlib import Path

import keras.backend as K
import numpy as np
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model
from sklearn.metrics import accuracy_score, fbeta_score, precision_recall_curve

from keras_efficientmixnets.custom_objects import Swish
from utils import load_data_set_from_pileups, setup_logger

K.clear_session()

""" 
Python file to quickly run evaluations for a model

"""

folder_name = "outputs/logs_" + str(date.today())
logging = setup_logger("eval_logger", "eval.log", folder_name)


def create_network(input_shape, keras_model, mixed=True):
    '''Conv Network architecture, based on a keras model such as EfficientNets
    '''
    inp = Input(shape=input_shape)
    base_model = keras_model(input_tensor=inp, weights=None, include_top=False, input_shape=input_shape, pooling="avg", mixed=mixed)
    x = base_model.output
    x = Dense(1024, use_bias=True)(x)
    x = Swish()(x)
    x = Dropout(0.5)(x)
    x = Dense(200, use_bias=True)(x)
    x = Swish()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(2, activation="softmax")(x)
    return Model(inp, predictions)

     

def eval_model(args):

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    with open(args.path_partition, 'r') as f:
        logging.info("Loading partition of the data...")
        partition = json.load(f)
        logging.info("Partition loaded...")
        
        
    logging.info("Predicting for : {}".format(args.chromosom))
    dics = [partition["paths_labels_" + chrom + "_down"] if args.is_downsampled else partition["paths_labels_" + chrom] for chrom in args.chromosom]
    
    paths_labels = {}
    for d in dics:
        paths_labels.update(d)
    
    
    X_test, y_test = load_data_set_from_pileups(paths_labels)

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
        model = create_network((299, 299, 5), getattr(module, args.architecture), mixed=args.mixed) # calls EfficientNetBX from a string 
        if args.path_model is not None:
            if Path(str(args.path_model)).exists():
                model.load_weights(args.path_model)
                logging.info("Pretrained weights loaded")
            else:
                raise ValueError("Path to pretrained weights doesn't exists !")

    logging.info("Predictions...")        
    y_pred = model.predict(X_test, verbose=1)
    logging.info("Prediction successful...")
    logging.info("Scoring predictions on {} images...".format(y_test.shape[0]))
  
    y_pred_prob = []
    for i in y_pred:
        if np.argmax(i) == 0:
            y_pred_prob.append(1 - i[0])
        else:
            y_pred_prob.append(i[np.argmax(i)])

    y_pred_prob = np.array(y_pred_prob)
    y_test_bin  = np.argmax(y_test, axis=1)

    precision, recall, tr = precision_recall_curve(y_test_bin, y_pred_prob)
    indx = np.where((args.min_precision <= precision) & (precision < args.max_precision))[0]
    
    m = 0
    threshold = 0
    min_recall = 0.9
    indx_opti = 0
    for i in indx:
        if recall[i] > min_recall and recall[i] > m:
            m = recall[i]
            threshold = tr[i]
            indx_opti = i
    
    logging.info("\naccuracy_score : {}".format(accuracy_score(y_test_bin, y_pred_prob >= threshold)))
    logging.info("\nPrecision : {}".format(precision[indx_opti]))
    logging.info("\nRecall : {}".format(recall[indx_opti]))
    logging.info("\nf_{} score : {}".format(args.beta, fbeta_score(y_test_bin, (y_pred_prob >= threshold), beta=args.beta)))
    
    if args.save_pred:
        filename = "_".join(["y_pred", args.architecture, str(date.today())])
        np.save(filename, y_pred)
 

if __name__ == "__main__":
    # execute only if run as a script
        
    parser = argparse.ArgumentParser(description='Script for evaluating predictions')
    parser.add_argument('-p', '--partition', dest='path_partition', required=True, type=str, help='path to partition containing list of index for test images')
    
    parser.add_argument('--downsample', dest='is_downsampled', action='store_true', help='whether to use less training images or not')
    parser.add_argument('--no_downsample', dest='is_downsampled', action='store_false', help='whether to use less training images or not')
    
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

    parser.add_argument('--max_precision', default=1, type=int, help='Maximum precision to consider when looking at precision-recall curve')
    parser.add_argument('--min_precision', default=0.9, type=int, help='Minimum precision to consider when looking at precision-recall curve')
    parser.add_argument('--beta', default=2, type=int, help='Order of f_beta score, beta=1 is regular f1 score, f2 score gives more importance to recall')
    parser.add_argument('--gpu', dest='gpu_id', default='1', type=str, help='ID of the GPU to use')

    parser.add_argument('--save_predict', dest='save_pred', action='store_true', help='whether to save predictions or not')
    parser.add_argument('--no_save_predict', dest='save_pred', action='store_false', help='whether to save predictions or not')

    parser.set_defaults(is_downsampled=True, mixed=True, save_pred=True)

    args = parser.parse_args()
    logging.info("Script called with args : {}".format(args))
    ### REM : C'est mieux de logger directement ce que l'utilisateur à taper comme commande
    ###  (avec sys.argv je crois), car là s'il y a qqch qui ne va pas, tu ne sauras pas
    ###  si c'est l'utilisateur qui n'a pas rentré un truc correct ou si c'est le parser
    ###  qui merde
    
    eval_model(args)

    ### REM : Une autre facon de faire que via les arguments en ligne de commande est de pouvoir
    ###     raisonner sur un fichier de paramètres (ou de configuration)
    ###     qui contient des lignes du type
    ###     nom_variable=qqch
    ###     permettant de décrire l'ensemble des valeurs des variables/paramètres
    ###     pour lancer ton code



