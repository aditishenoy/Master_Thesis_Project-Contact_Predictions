#This file contains code where unet++ architecture has been implemented
#for a regression model using sigmoid distribution for distance output
# i.e. out_sccore has been changed to out_dist as opposed to original regression model

from __future__ import division

"-------------------------------------------------------------------"
#Calling libraries and initialization of variables
import os
import sys
import h5py
import numpy as np

from sklearn.externals import joblib
from sklearn import preprocessing
from collections import defaultdict, namedtuple

import keras
import keras.backend as K
from keras.regularizers import l2
from keras.layers import Activation
from keras.layers.core import Lambda
from keras.models import Model, load_model
from keras.utils import np_utils, plot_model
from keras.layers.merge import concatenate
from keras.layers import Input, Dropout, BatchNormalization
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers import Conv2D, Conv1D, MaxPooling2D, UpSampling2D

DROPOUT = 0.1
ACTIVATION = ELU
INIT = "he_normal"

#reg_strength = float(sys.argv[2])
reg_strength = float(10**-12)
REG = l2(reg_strength)

"-------------------------------------------------------------------"
#Definition of functions needed for u-net architecture

def self_outer(x):
    outer_x = x[ :, :, None, :] * x[ :, None, :, :]
    return outer_x

def add_2D_conv(model, filters, kernel_size, data_format="channels_last", padding="same", 
        depthwise_initializer=INIT, pointwise_initializer=INIT, depthwise_regularizer=REG, 
        pointwise_regularizer=REG, separable=True, namesuffix=""):

    if separable:
        raise ValueError('Separable!')
    
    if namesuffix:
        model = Conv2D(filters, kernel_size, data_format = data_format, padding = padding, 
                kernel_initializer = depthwise_initializer, kernel_regularizer = depthwise_regularizer, 
                name = "separable_conv2d_" + namesuffix)(model)

        model = ACTIVATION(name="activation_" + namesuffix)(model)
        model = BatchNormalization(name="batch_normalization_" + namesuffix)(model)
        model = Dropout(DROPOUT, name="dropout_" + namesuffix)(model)
    else:
        model = Conv2D(filters, kernel_size, data_format=data_format, padding=padding, kernel_initializer=depthwise_initializer, kernel_regularizer=depthwise_regularizer)(model)
        model = ACTIVATION()(model)
        model = BatchNormalization()(model)
        model = Dropout(DROPOUT)(model)
    return model


def _add_binary_head(model, dist_cutoff, kernel_size):
    
    out_binary = Conv2D(1, kernel_size, activation = "sigmoid", 
            data_format = "channels_last", padding = "same", kernel_initializer = INIT, 
            kernel_regularizer = REG, name = "out_binary_%s" % dist_cutoff)(model)
    
    return out_binary

"-------------------------------------------------------------------"
#Wrap the model (Use input_2D and input_seq as inputs produce out_dist and out_binary as outputs)

def wrap_model(model, binary_cutoffs):

    # inputs for sequence features
    inputs_seq = [Input(shape=(None, 22), dtype=K.floatx(), name="seq"), # sequence
                  Input(shape=(None, 23), dtype=K.floatx(), name="self_info"), # self-information
                  Input(shape=(None, 23), dtype=K.floatx(), name="part_entr")] # partial entropy

    # input for 2D features
    #inputs_2D = [Input(shape=(None, None, 1), name="plm", dtype=K.floatx()), # plm
    inputs_2D = [Input(shape=(None, None, 1), name="gdca", dtype=K.floatx()), # gdca
                 Input(shape=(None, None, 1), name="mi_corr", dtype=K.floatx()), # mi_corr
                 Input(shape=(None, None, 1), name="nmi_corr", dtype=K.floatx()), # nmi_corr
                 Input(shape=(None, None, 1), name="cross_h", dtype=K.floatx())] # cross_h

    # input for masking missing residues
    input_mask = Input(shape=(None, None, 1), name="mask")

    out_lst = model(inputs_2D + inputs_seq)
    out_mask_lst = []
    out_names = ["out_dist_mask"] + ["out_binary_%s_mask" % d for d in binary_cutoffs]

    for i, out in enumerate(out_lst):
        if out_names[i] != "out_dist_mask":
            out = keras.layers.Multiply(name = out_names[i])([out, input_mask])
        else:
            #out = keras.layers.Lambda(lambda x: x + 0, input_shape=(None, None, 12), name="out_dist_mask")(out)
            out = keras.layers.Dropout(0, name="out_dist_mask")(out)
        out_mask_lst.append(out)

    wrapped_model = Model(inputs=inputs_2D + inputs_seq + [input_mask], outputs=out_mask_lst)

    return wrapped_model

"-------------------------------------------------------------------"
#Creating U-net++ architecture

def create_unet_plus(filters=64,
                ss_model_path = "/home/ashenoy/ashenoy/david_retrain_pconsc4/models/ss_pred_resnet_elu_nolr_dropout01_l26_large_v3_saved_model.h5", binary_cutoffs=[]):

    inputs_seq = [Input(shape=(None, 22), dtype=K.floatx()), # sequence
                  Input(shape=(None, 23), dtype=K.floatx()), # self-information
                  Input(shape=(None, 23), dtype=K.floatx())] # partial entropy

    ss_model = load_model(ss_model_path)
    ss_model.trainable = False

    seq_feature_model = ss_model._layers_by_depth[5][0]
    #plot_model(seq_feature_model, "seq_feature_model.png")
    assert 'model' in seq_feature_model.name, seq_feature_model.name
    seq_feature_model.name = 'sequence_features'
    seq_feature_model.trainable = False
    for l in ss_model.layers:
        l.trainable = False
    for l in seq_feature_model.layers:
        l.trainable = False

    bottleneck_seq = seq_feature_model(inputs_seq)
    model_1D_outer = Lambda(self_outer)(bottleneck_seq)
    model_1D_outer = BatchNormalization()(model_1D_outer)

    inputs_2D = [Input(shape=(None, None, 1), dtype=K.floatx()), # plm/gdca
                 Input(shape=(None, None, 1), dtype=K.floatx()), # mi_corr
                 Input(shape=(None, None, 1), dtype=K.floatx()), # nmi_corr
                 Input(shape=(None, None, 1), dtype=K.floatx())] # cross_h


    unet = keras.layers.concatenate(inputs_2D + [model_1D_outer])

    unet = add_2D_conv(unet, filters, 1, separable=False)
    unet = add_2D_conv(unet, filters, 3, separable=False)
    unet = add_2D_conv(unet, filters, 3, separable=False)
    link1 = unet
    unet = MaxPooling2D()(unet)
    unet = add_2D_conv(unet, filters*2, 3, separable=False)
    unet = add_2D_conv(unet, filters*2, 3, separable=False)
    link2 = unet
    unet = MaxPooling2D()(unet)
    unet = add_2D_conv(unet, filters*4, 3, separable=False)
    unet = add_2D_conv(unet, filters*4, 3, separable=False)
    link3 = unet
    unet = MaxPooling2D()(unet)
    unet = add_2D_conv(unet, filters*8, 3, separable=False)
    unet = add_2D_conv(unet, filters*8, 3, separable=False)
    link4 = unet
    unet = MaxPooling2D()(unet)
    unet = add_2D_conv(unet, filters*16, 3, separable=False)
    unet = add_2D_conv(unet, filters*16, 3, separable=False)

    unet = UpSampling2D()(unet)
    unet = add_2D_conv(unet, filters*8, 2, separable=False)
    unet = keras.layers.concatenate([unet, link4])
    unet = add_2D_conv(unet, filters*8, 3, separable=False)
    unet = add_2D_conv(unet, filters*8, 3, separable=False)
    unet = UpSampling2D()(unet)
    unet = add_2D_conv(unet, filters*4, 2, separable=False)
    unet = keras.layers.concatenate([unet, link3])
    unet = add_2D_conv(unet, filters*4, 3, separable=False)
    unet = add_2D_conv(unet, filters*4, 3, separable=False)
    unet = UpSampling2D()(unet)
    unet = add_2D_conv(unet, filters*2, 2, separable=False)
    unet = keras.layers.concatenate([unet, link2])
    unet = add_2D_conv(unet, filters*2, 3, separable=False)
    unet = add_2D_conv(unet, filters*2, 3, separable=False)
    unet = UpSampling2D()(unet)
    unet = add_2D_conv(unet, filters, 2, separable=False)
    unet = keras.layers.concatenate([unet, link1])
    split = unet
    unet = add_2D_conv(unet, filters, 3, separable=False)
    unet = add_2D_conv(unet, filters, 3, separable=False)

    out_binary_lst = []
    if binary_cutoffs:
        for d in binary_cutoffs:
            out_binary_lst.append(add_binary_head(unet, d, 7, filters))

    unet = add_2D_conv(split, filters, 3, separable=False)
    unet = add_2D_conv(unet, filters, 3, separable=False)

    out_dist = Conv2D(1, 7, activation="sigmoid", data_format="channels_last", padding="same",
                        kernel_initializer=INIT,
                        kernel_regularizer=out_reg,
                        name="out_dist")(unet)

    model = Model(inputs=inputs_2D + inputs_seq, outputs=[out_dist] + out_binary_lst)
    return model



if __name__ == "__main__":

    modelfile = sys.argv[1]
    suffix = os.path.splitext(modelfile)[0]
    #num_blocks = int(sys.argv[2])

    binary_cutoffs = [6, 8, 10]

    model = create_unet_plus(binary_cutoffs=binary_cutoffs)
    plot_model(model, "unet.png")
    print(model.summary())
    model = wrap_model(model, binary_cutoffs)
    model.save(modelfile)

