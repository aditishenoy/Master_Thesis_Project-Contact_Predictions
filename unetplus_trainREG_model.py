from __future__ import division

import sys
import os
import random
from collections import defaultdict

import h5py
import numpy as np
from numpy import log10

import keras
import keras.backend as K
from keras.regularizers import l2
from keras.models import Model, load_model
from keras.layers import Input, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, Conv1D
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers.merge import Concatenate
#from keras.utils import np_utils, plot_model
from keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint, TensorBoard

DROPOUT = 0.1
ACTIVATION = ELU
INIT = "he_normal"
REG = None


def generator_from_file(h5file, feat_lst, label, binary_cutoffs, batch_size=1):
    key_lst = list(h5file[label].keys())
    random.shuffle(key_lst)
    i = 0

    while True:
        # TODO: different batch sizes with padding to max(L)
        # for i in range(batch_size):
        # index = random.randint(1, len(features)-1)
        if i == len(key_lst):
            random.shuffle(key_lst)
            i = 0

        key = key_lst[i]

        x_i_dict, mask, y, y_binary_dict = get_datapoint(h5file, feat_lst, label, binary_cutoffs, key)

        batch_features_dict = x_i_dict
        batch_features_dict["mask"] = mask

        batch_labels_dict = {}
        batch_labels_dict["out_%s_mask" % label] = y
        for d, y_binary in y_binary_dict.items():
            batch_labels_dict["out_binary_%s_mask" % d] = y_binary

        i += 1

        yield batch_features_dict, batch_labels_dict


def generator(features_dict, labels_dict, batch_size=1):
    index_lst = list(range(len(features_dict[list(features_dict.keys())[0]])))
    random.shuffle(index_lst)
    i = 0

    while True:
        # TODO: different batch sizes with padding to max(L)
        # for i in range(batch_size):
        # index = random.randint(1, len(features)-1)
        if i == len(index_lst):
            random.shuffle(index_lst)
            i = 0

        index = index_lst[i]

        batch_features_dict = {}
        for key, feature_lst in features_dict.items():
            batch_features_dict[key] = feature_lst[index]

        batch_labels_dict = {}
        for key, label_lst in labels_dict.items():
            batch_labels_dict[key] = label_lst[index]

        i += 1

        yield batch_features_dict, batch_labels_dict


def get_test_batch(features_dict, labels_dict, id_lst, batch_size=1):
    for index, acc in enumerate(id_lst):
        if acc != "1AHSC.hhE0":
            continue

        batch_features_dict = {}
        for key, feature_lst in features_dict.items():
            batch_features_dict[key] = feature_lst[index]

        batch_labels_dict = {}
        for key, label_lst in labels_dict.items():
            batch_labels_dict[key] = label_lst[index]

        return batch_features_dict, batch_labels_dict


def pad(x, pad_even, depth=4):
    divisor = np.power(2, depth)
    remainder = x.shape[0] % divisor
    # no padding
    if not pad_even:
        return x
    # no padding because already of even shape
    elif pad_even and remainder == 0:
        return x
    # add zero rows after 1D feature
    elif len(x.shape) == 2:
        return np.pad(x, [(0, divisor - remainder), (0, 0)], "constant")
    # add zero columns and rows after 2D feature
    elif len(x.shape) == 3:
        return np.pad(x, [(0, divisor - remainder), (0, divisor - remainder), (0, 0)], "constant")


def get_datapoint(h5file, feat_lst, label, binary_cutoffs, key, pad_even=False):
    x_i_dict = {}
    for feat in feat_lst:
        if feat in ['sep']:
            x_i = np.array(range(L))
            x_i = np.abs(np.add.outer(x_i, -x_i))
        elif feat in ['gneff']:
            x_i = h5file[feat][key][()]
            x_i = log10(x_i)
            # x_i = np.outer(x_i, x_i)
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
    mask = mask[..., None]  # reshape from (L,L) to (L,L,1)
    mask = pad(mask, pad_even)
    mask = mask[None, ...]  # reshape from (L,L,1) to (1,L,L,1)

    y = h5file[label][key][()]
    y = y[..., None]  # reshape from (L,L) to (L,L,1)
    y = pad(y, pad_even)
    y = y[None, ...]

    y_binary_dict = {}
    y_dist = h5file["dist"][key][()]
    for d in binary_cutoffs:
        y_binary = y_dist < d
        y_binary = y_binary[..., None]  # reshape from (L,L) to (1,L,L,1)
        y_binary = pad(y_binary, pad_even)
        y_binary = y_binary[None, ...]
        y_binary_dict[d] = y_binary

    return x_i_dict, mask, y, y_binary_dict, L


def get_data(h5file, feat_lst, label, binary_cutoffs, pad_even=False, val_id_lst=[]):
    data_x_dict = defaultdict(list)
    data_y_dict = defaultdict(list)
    val_x_dict = defaultdict(list)
    val_y_dict = defaultdict(list)
   
    id_lst = []
    len_dict = []
    #assert  h5file[label].keys() == h5file[feat_lst[0]].keys()
    key_lst = list(h5file[label].keys())
    
    for key in key_lst:
        x_i_dict, mask, y, y_binary_dict, L = get_datapoint(h5file, feat_lst, label, binary_cutoffs, key,
                                                            pad_even=pad_even)
        if key in val_id_lst:
            for feat, x_i in x_i_dict.items():
                val_x_dict[feat].append(x_i)
            val_x_dict["mask"].append(mask)
            val_y_dict["out_%s_mask" % label].append(y)
            for d, y_binary in y_binary_dict.items():
                val_y_dict["out_binary_%s_mask" % d].append(y_binary)
        else:
            for feat, x_i in x_i_dict.items():
                data_x_dict[feat].append(x_i)
            data_x_dict["mask"].append(mask)
            data_y_dict["out_%s_mask" % label].append(y)
            for d, y_binary in y_binary_dict.items():
                data_y_dict["out_binary_%s_mask" % d].append(y_binary)
        id_lst.append(key)

    return data_x_dict, data_y_dict, val_x_dict, val_y_dict, id_lst


def train(infile, modelfile, suffix="", feat_lst=[], binary_cutoffs=[], test_data_file="",
          val_id_file="/home/ashenoy/ashenoy/david_retrain_pconsc4/IDs_training_pdbcull_170914_A_before160501_validation.txt"):
    f = h5py.File(infile, "r")
    assert (len(f['dist']) == len(f['plm']) == len(f['sscore']))
    idlist = f['sscore'].keys()

    label = "sscore"
    
    model = load_model(modelfile)
    #plot_model(model, "model.%s.png" % suffix)

    loss_dict = {"out_sscore_mask": keras.losses.mean_absolute_error}
    #loss_dict =  {"out_sscore_mask": keras.losses.mean_squared_error}
    loss_weight_dict = {"out_sscore_mask": 1}
    for d in binary_cutoffs:
        loss_dict["out_binary_%s_mask" % d] = keras.losses.binary_crossentropy
        loss_weight_dict["out_binary_%s_mask" % d] = 1. / len(binary_cutoffs)

    model.compile(loss=loss_dict, loss_weights=loss_weight_dict, optimizer=keras.optimizers.Adam(),
                  metrics=['mae', 'mse', 'kullback_leibler_divergence'])


    ### training parameters

    batch_size = 1
    epochs = 100
    verbose = 1
    #print (val_id_file)
    with open(val_id_file) as val_id_f:
        val_id_lst = val_id_f.read().splitlines()
    
    x_train_dict, y_train_dict, x_val_dict, y_val_dict, id_lst = get_data(f, feat_lst, label, binary_cutoffs,
                                                                          pad_even=True, val_id_lst=val_id_lst)
    
    #print (len(x_train_dict))
    #print (len(y_train_dict))
    #print (len(x_val_dict))
    #print (len(y_val_dict))

    num_steps = len(y_train_dict["out_sscore_mask"])
    num_steps_val = len(y_val_dict["out_sscore_mask"])
   
    print(num_steps)
    print(num_steps_val)

    training_generator = generator(x_train_dict, y_train_dict)
    validation_generator = generator(x_val_dict, y_val_dict)


    
    # training_generator = generator_from_file(f, feat_lst, label, binary_cutoffs)

    test_data = None
    if test_data_file:
        # if False:
        test_f = h5py.File(test_data_file, "r")
        x_test_dict, y_test_dict, _, _, id_lst_test = get_data(test_f, feat_lst, label, binary_cutoffs, pad_even=True)
        test_data = get_test_batch(x_test_dict, y_test_dict, id_lst_test)

    ### callbacks
    if binary_cutoffs:
        reduce_lr = ReduceLROnPlateau(monitor='out_sscore_mask_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
        csv_logger = CSVLogger("%s.log" % suffix)
        if not os.path.exists(suffix):
            os.makedirs(suffix)
        model_path = "models/%s/%s_epo{epoch:02d}-{out_sscore_mask_loss:.4f}.h5" % (suffix, suffix)
        checkpoint = ModelCheckpoint(model_path, monitor='out_sscore_mask_loss', verbose=1, save_best_only=False,
                                     mode='min')
    else:
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
        csv_logger = CSVLogger("%s.log" % suffix)
        if not os.path.exists(suffix):
            os.makedirs(suffix)
        model_path = "models/%s/%s_epo{epoch:02d}-{loss:.4f}.h5" % (suffix, suffix)
        checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=False, mode='min')

    # tb = TensorBoard(log_dir="%s/" % suffix, histogram_freq=1, batch_size=1, write_images=True)
    tb = TensorBoard(log_dir="models/%s/" % suffix, batch_size=1, write_images=True)

    #print(model.summary())
    
    
    #model.fit_generator(training_generator, num_steps, epochs=epochs, verbose=verbose, validation_data=test_data, use_multiprocessing=True, callbacks=[reduce_lr, csv_logger, checkpoint, tb])
    
    
    model.fit_generator(training_generator, num_steps, epochs=epochs, verbose=verbose, validation_data=validation_generator, validation_steps=num_steps_val, use_multiprocessing=True, callbacks=[reduce_lr, csv_logger, checkpoint, tb])
    
    model.trainable=False
    model.save('%s_mae_trained.h5' % suffix, include_optimizer=False)
    #unet.save('%s_unet_trained.h5' % suffix, include_optimizer=False)
    return model


def ppv(x, y, f=1.0, sep=5):
    x_u = np.triu(x, k=sep)
    y_u = np.triu(x, k=sep) > 8


def cmap_to_list(cmap):
    # make matrix symmetric
    cmap = (cmap + cmap.T) / 2.
    result = []
    for (i, j), val in np.ndenumerate(cmap):
        if i <= j:
            result.append((val, i + 1, j + 1))
    return result


if __name__ == "__main__":
    train_data = '/home/ashenoy/ashenoy/david_retrain_pconsc4/training_pdbcull_170914_A_before160501.h5'
    test_data = '/home/ashenoy/ashenoy/david_retrain_pconsc4/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5'

    # suffix = ".ali.plm-gdca-phy-ss-rsa"
    modelfile = sys.argv[1]
    suffix = os.path.splitext(modelfile)[0]
    testing = False

    # always using plm, gdca, ss, rsa
    # + abbreviations: seq=s, self_info=i, nmi_corr=n, cross_h=c, gneff=g, part_entr=p
    # feat_lst = ['plm', 'gdca', 'phy', 'ss', 'rsa', 'gneff']
    # feat_lst = ['plm', 'gdca', 'ss', 'rsa', 'gneff', 'self_info', 'part_entr']
    # feat_lst = ['plm', 'gdca', 'ss', 'rsa', 'gneff', 'part_entr', 'nmi_corr', 'cross_h']
    # feat_lst = ['plm_J', 'cross_h', 'nmi_corr', 'mi_corr', 'seq', 'part_entr', 'self_info']
    # feat_lst = ['plm', 'cross_h', 'nmi_corr', 'mi_corr', 'seq', 'part_entr', 'self_info']
    feat_lst = ['gdca', 'cross_h', 'nmi_corr', 'mi_corr', 'seq', 'part_entr', 'self_info']
    # feat_lst = ['plm', 'gdca', 'ss', 'rsa', 'seq', 'cross_h'] # used in resnetX_multi-8-12_1Dseq_regY runs
    # feat_lst = ['plm']

    binary_cutoffs = [6, 8, 10]

    model = train(train_data, modelfile, suffix=suffix, feat_lst=feat_lst, binary_cutoffs=binary_cutoffs,
                  test_data_file=test_data, val_id_file='/home/ashenoy/ashenoy/david_retrain_pconsc4/IDs_training_pdbcull_170914_A_before160501_validation.txt')

