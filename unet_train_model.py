from __future__ import division

"-------------------------------------------------------------------"
#Calling libraries and initialization of variables

import sys
import os
import random
from collections import defaultdict

import h5py
import numpy as np
from numpy import log10

import keras
import keras.backend as K
from keras import losses
from keras.models import Model, load_model
from keras.utils import to_categorical 
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard

"-------------------------------------------------------------------"
#Generate dictionary of features and labels

def generator_from_file(h5file, feat_lst, label, binary_cutoffs, batch_size = 1):    
   
    key_lst = list(h5file[label].keys())
    random.shuffle(key_lst)
    i = 0

    key = key_lst[i]

    x_i_dict, mask, y, y_binary_dict, L = get_datapoint(h5file, feat_lst, label, binary_cutoffs, key)
    
    bins = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
 
    batch_features_dict = x_i_dict
    batch_features_dict["mask"] = mask

    batch_labels_dict = {}
    y = np.searchsorted(bins, y)
    y = to_categorical(y, num_classes = 12)
    batch_labels_dict["out_%s_mask" % label] = y

    for d, y_binary in y_binary_dict.items():
        batch_labels_dict["out_binary_%s_mask" % d] = y_binary

    i += 1

    yield batch_features_dict, batch_labels_dict


def get_datapoint(h5file, feat_lst, label, binary_cutoffs, key, pad_even = False):

    x_i_dict = {}
    
    for feat in feat_lst:
        if feat in ['sep']:
            x_i = np.array(range(L))
            x_i = np.abs(np.add.outer(x_i, -x_i))
        elif feat in ['gneff']:
            x_i = h5file[feat][key][()]
            x_i = log10(x_i)
        elif feat in ['plm_J']:
            x_i = h5file[feat][key][()]
            L = x_i.shape[0]
        elif feat in ['cross_h', 'mi_corr', 'nmi_corr', 'plm', 'gdca']:
            x_i = h5file[feat][key][()]
            L = x_i.shape[0]
            x_i = x_i[..., None]
        else:
            x_i = h5file[feat][key][()]
            L = x_i.shape[0]
        x_i = pad(x_i, pad_even)
        x_i_dict[feat] = x_i[None, ...]

    mask = h5file[label][key][()]
    mask = mask != 0.0
    mask = mask[..., None]
    mask = pad(mask, pad_even)
    mask = mask[None, ...]

    y = h5file[label][key][()]
    y = y[..., None]
    y = pad(y, pad_even)
    y = y[None, ...]

    y_binary_dict = {}
    y_dist = h5file["dist"][key][()]

    for d in binary_cutoffs:
        y_binary = y_dist < d
        y_binary = y_binary[..., None]
        y_binary = pad(y_binary, pad_even)
        y_binary = y_binary[None, ...]
        y_binary_dict[d] = y_binary

    return x_i_dict, mask, y, y_binary_dict, L


def pad(x, pad_even, depth = 4):

    divisor = np.power(2, depth)
    remainder = x.shape[0] % divisor

    if not pad_even:
        return x
    elif pad_even and remainder == 0:
        return x
    elif len(x.shape) == 2:
        return np.pad(x, [(0, divisor - remainder), (0,0)], "constant")
    elif len(x.shape) == 3:
        return np.pad(x, [(0, divisor - remainder), (0, divisor - remainder), (0,0)], "constant")

"-------------------------------------------------------------------"



"-------------------------------------------------------------------"
"-------------------------------------------------------------------"
"-------------------------------------------------------------------"
"-------------------------------------------------------------------"
"-------------------------------------------------------------------"
"-------------------------------------------------------------------"
#Main function call

if __name__ == "__main__":

    train_data = '/home/ashenoy/ashenoy/david_retrain_pconsc4/training_pdbcull_170914_A_before160501.h5' 
    test_data = '/home/ashenoy/ashenoy/david_retrain_pconsc4/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5'

    modelfile = sys.argv[1]

    suffix = os.path.splitext(modelfile)[0]
    testing = False

    feat_lst = ['gdca', 'cross_h', 'nmi_corr', 'mi_corr', 'seq', 'part_entr', 'self_info']

    binary_cutoffs = [6, 8, 10]

    "----------"

    f = h5py.File(train_data, "r")

    gen = generator_from_file(f, feat_lst,  label = "dist", binary_cutoffs = binary_cutoffs)
    feats, labels = next(gen)

    for k, v in labels.items():
        print (k, v.shape)
        if 'dist' in k:
            print (v.shape)
