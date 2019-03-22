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
#Generate dictionary of features and labels from training file 

def generator_from_file(h5file, feat_lst, label, binary_cutoffs, batch_size = 1):    
   
    key_lst = list(h5file[label].keys())
    random.shuffle(key_lst)
    i = 0

    key = key_lst[i]

    x_i_dict, mask, y, y_binary_dict, L = get_datapoint(h5file, feat_lst, label, binary_cutoffs, key)
    
    #bins = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16]
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

"-------------------------------------------------------------------"
#Functions for training the model (get_datapoint, pad, get_data, generator, get_test_batch)

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

def get_data(h5file, feat_lst, label, binary_cutoffs, pad_even = False, val_id_lst = []):

    data_x_dict = defaultdict(list)
    data_y_dict = defaultdict(list)
    val_x_dict = defaultdict(list)
    val_y_dict = defaultdict(list)

    id_lst = []
    len_dict = []
    #bins = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16]
    bins = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    assert h5file[label].keys() == h5file[feat_lst[0]].keys()

    key_lst = list(h5file[label].keys())

    for key in key_lst:
        x_i_dict, mask, y, y_binary_dict, L = get_datapoint(h5file, feat_lst, label, binary_cutoffs, key, pad_even = pad_even)
        
        if key in val_id_lst:
            for feat, x_i in x_i_dict.items():
                val_x_dict[feat].append(x_i)
            val_x_dict["mask"].append(mask)
            y = np.searchsorted(bins, y)
            y = to_categorical(y, num_classes = 12)
            
            val_y_dict["out_%s_mask" % label].append(y)
            
            for d, y_binary in y_binary_dict.items():
                val_y_dict["out_binary_%s_mask" % d].append(y_binary)
        else:
            for feat, x_i in x_i_dict.items():
                data_x_dict[feat].append(x_i)
            data_x_dict["mask"].append(mask)
            y = np.searchsorted(bins, y)
            y = to_categorical(y, num_classes = 12)
            data_y_dict["out_%s_mask" % label].append(y)
            for d, y_binary in y_binary_dict.items():
                data_y_dict["out_binary_%s_mask" % d].append(y_binary)
        id_lst.append(key)

    return data_x_dict, data_y_dict, val_x_dict, val_y_dict, id_lst


def generator(features_dict, labels_dict, batch_size = 1):

    index_lst = list(range(len(features_dict[list(features_dict.keys())[0]])))
    random.shuffle(index_lst)
    i = 0
    #bins = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16]
    #bins = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    while True:

        if i == len(index_lst):
            random.shuffle(index_lst)
            i = 0

        index = index_lst[i]

        batch_features_dict = {}
        for key, feature_lst in features_dict.items():
            batch_features_dict[key] = feature_lst[index]

        batch_labels_dict = {}
        for key, labels_lst in labels_dict.items():
            batch_labels_dict[key] = labels_lst[index]

        i += 1

        yield batch_features_dict, batch_labels_dict

def get_test_batch(features_dict, labels_dict, id_lst, batch_size = 1):
    for index, acc in enumerate(id_lst):
        if acc != "1AHSC.hhE0":
            continue

        batch_features_dict = {}
        for key, features_lst in features_dict.items():
            batch_features_dict[key] = features_lst[index]

        batch_labels_dict = {}
        for key, labels_lst in labels_dict.items():
            batch_labels_dict[key] = labels_lst[index]

        return batch_features_dict, batch_labels_dict

"-------------------------------------------------------------------"
#Training the model

def train(infile, modelfile, suffix = "", 
        feat_lst = [], binary_cutoffs = [], test_data_file = "", 
        val_id_file="/home/ashenoy/ashenoy/aditi_retrain_pconsc4/IDs_training_pdbcull_170914_A_before160501_validation.txt"): 
    
    f = h5py.File(infile, "r")
    
    assert (len(f['dist']) == len(f['plm']) == len(f['sscore']))
    idlist = f['dist'].keys()

    label = "dist"
    
    model = load_model(modelfile)

    loss_dict = {"out_dist_mask" : keras.losses.categorical_crossentropy}
    loss_weight_dict = {"out_dist_mask": 1}

    for d in binary_cutoffs:
        loss_dict["out_binary_%s_mask" %d] = keras.losses.binary_crossentropy
        loss_weight_dict["out_binary_%s_mask" %d] = 1./len(binary_cutoffs)

    model.compile(loss = loss_dict, loss_weights = loss_weight_dict, optimizer = keras.optimizers.Adam(), metrics = ['mae', 'mse'])

    batch_size = 1
    epochs = 100
    verbose = 1

    with open(val_id_file) as val_id_f:
        val_id_lst = val_id_f.read().splitlines()
    
    x_train_dict, y_train_dict, x_val_dict, y_val_dict, id_lst = get_data(f, feat_lst, label, binary_cutoffs,
                                                                          pad_even=True, val_id_lst=val_id_lst)
    
    num_steps = len(y_train_dict["out_dist_mask"])
    num_steps_val = len(y_val_dict["out_dist_mask"])

    print(num_steps)
    print(num_steps_val)

    validation_generator = generator(x_val_dict, y_val_dict)
    training_generator = generator(x_train_dict, y_train_dict)
    
    feats, labels = next(validation_generator)
    for k, v in labels.items():
        print (k, v.shape)
    test_data = None
    
    if test_data_file:
        test_f = h5py.File(test_data_file, "r")
        x_test_dict, y_test_dict, _, _, id_lst_test = get_data(test_f, feat_lst, label, binary_cutoffs, pad_even=True)
        test_data = get_test_batch(x_test_dict, y_test_dict, id_lst_test)
    
    if binary_cutoffs:
        reduce_lr = ReduceLROnPlateau(monitor='out_dist_mask_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
        csv_logger = CSVLogger("%s.log" % suffix)
        if not os.path.exists(suffix):
            os.makedirs(suffix)
        model_path = "models/%s/%s_epo{epoch:02d}-{out_dist_mask_loss:.4f}.h5" % (suffix, suffix)
        checkpoint = ModelCheckpoint(model_path, monitor='out_dist_mask_loss', verbose=1, save_best_only=False,
                                     mode='min')
    else:
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
        csv_logger = CSVLogger("%s.log" % suffix)
        if not os.path.exists(suffix):
            os.makedirs(suffix)
        model_path = "models/%s/%s_epo{epoch:02d}-{loss:.4f}.h5" % (suffix, suffix)
        checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=False, mode='min')

    tb = TensorBoard(log_dir="models/%s/" % suffix, batch_size=1, write_images=True)

    model.fit_generator(training_generator, num_steps, epochs=epochs, verbose=verbose,
                        validation_data=validation_generator, validation_steps=num_steps_val, use_multiprocessing=True,
                        callbacks=[reduce_lr, csv_logger, checkpoint, tb])
    model.trainable=False
    model.save('%s_mae_trained.h5' % suffix, include_optimizer=False)
    
    return model

"-------------------------------------------------------------------"
#Main function call

if __name__ == "__main__":

    train_data = '/home/ashenoy/ashenoy/david_retrain_pconsc4/training_pdbcull_170914_A_before160501.h5' 
    test_data = '/home/ashenoy/ashenoy/david_retrain_pconsc4/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5'
    val_id_file='/home/ashenoy/ashenoy/aditi_retrain_pconsc4/IDs_training_pdbcull_170914_A_before160501_validation.txt'
    
    modelfile = sys.argv[1]

    suffix = os.path.splitext(modelfile)[0]
    testing = False

    feat_lst = ['gdca', 'cross_h', 'nmi_corr', 'mi_corr', 'seq', 'part_entr', 'self_info']

    binary_cutoffs = [6, 8, 10]

    f = h5py.File(train_data, "r")

    model = train(train_data, modelfile, suffix=suffix, feat_lst=feat_lst, binary_cutoffs=binary_cutoffs,
                  test_data_file=test_data, val_id_file = val_id_file)


"-------------------------------------------------------------------"

