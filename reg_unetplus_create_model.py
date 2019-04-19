# This is a script for regression problem
# This file contains code where unet++ architecture has been implemented
# for a regression model using sigmoid distribution for S-score output

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
from keras.layers.convolutional import Deconv2D as Conv2DTranspose

DROPOUT = 0.1
smooth = 1.
ACTIVATION = ELU
INIT = "he_normal"

#reg_strength = float(sys.argv[2])
reg_strength = float(10**-12)
REG = l2(reg_strength)

"-------------------------------------------------------------------"
#Definition of functions needed for U-net++ architecture

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
        model = Dropout(DROPOUT, name="dropout_" + namesuffix)(model)
        model = ACTIVATION(name="activation_" + namesuffix)(model)
        model = BatchNormalization(name="batch_normalization_" + namesuffix)(model)

        model = Conv2D(filters, kernel_size, data_format = data_format, padding = padding, 
                kernel_initializer = depthwise_initializer, kernel_regularizer = depthwise_regularizer, 
                name = "separable_conv2d_" + namesuffix)(model)
        model = Dropout(DROPOUT, name="dropout_" + namesuffix)(model)
        model = ACTIVATION(name="activation_" + namesuffix)(model)
        model = BatchNormalization(name="batch_normalization_" + namesuffix)(model)



    else:
        model = Conv2D(filters, kernel_size, data_format=data_format, padding=padding, kernel_initializer=depthwise_initializer, kernel_regularizer=depthwise_regularizer)(model)
        model = Dropout(DROPOUT)(model)        
        model = ACTIVATION()(model)
        model = BatchNormalization()(model)

        model = Conv2D(filters, kernel_size, data_format=data_format, padding=padding, kernel_initializer=depthwise_initializer, kernel_regularizer=depthwise_regularizer)(model)
        model = Dropout(DROPOUT)(model)        
        model = ACTIVATION()(model)
        model = BatchNormalization()(model)

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
    out_names = ["out_sscore_mask"] + ["out_binary_%s_mask" % d for d in binary_cutoffs]

    for i, out in enumerate(out_lst):
        if out_names[i] != "out_sscore_mask":
            out = keras.layers.Multiply(name = out_names[i])([out, input_mask])
        else:
            #out = keras.layers.Lambda(lambda x: x + 0, input_shape=(None, None, 12), name="out_dist_mask")(out)
            out = keras.layers.Dropout(0, name="out_sscore_mask")(out)
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

    #Downsampling
    
    unet = add_2D_conv(unet, filters, 1, separable=False)
    unet = add_2D_conv(unet, filters, 3, separable=False)

    conv1_1 = add_2D_conv(unet, filters, 3, separable=False)    
    pool1 = MaxPooling2D(data_format = "channels_last")(conv1_1)
    
    conv2_1 = add_2D_conv(pool1, filters*2, 3, separable=False)
    pool2 = MaxPooling2D(data_format = "channels_last")(conv2_1)
    
    up1_2 = Conv2DTranspose(filters, 3)(conv2_1)
    conv1_2 = keras.layers.concatenate([up1_2, conv1_1])
    conv1_2 = add_2D_conv(conv1_2, filters, 3, separable=False)
    
    conv3_1 = add_2D_conv(pool2, filters*4, 3, separable=False)
    pool3 = MaxPooling2D(data_format = "channels_last")(conv3_1)
    
    up2_2 = Conv2DTranspose(filters*2, 3)(conv3_1)
    conv2_2 = keras.layers.concatenate([up2_2, conv2_1])
    conv2_2 = add_2D_conv(conv2_2, filters*2, 3, separable=False)
    
    up1_3 = Conv2DTranspose(filters, 3)(conv2_2)
    conv1_3 = keras.layers.concatenate([up1_3, conv1_1, conv1_2])
    conv1_3 = add_2D_conv(conv1_3, filters, 3, separable=False)
    
    conv4_1 = add_2D_conv(pool3, filters*8, 3, separable=False)
    pool4 = MaxPooling2D(data_format = "channels_last")(conv4_1)

    up3_2 = Conv2DTranspose(filters*4, 3)(conv4_1)
    conv3_2 = keras.layers.concatenate([up3_2, conv3_1])
    conv3_2 = add_2D_conv(conv3_2, filters*4, 3, separable=False)

    up2_3 = Conv2DTranspose(filters*2, 3)(conv3_2)
    conv2_3 = keras.layers.concatenate([up2_3, conv2_1, conv2_2])
    conv2_3 = add_2D_conv(conv2_3, filters*2, 3, separable=False)

    up1_4 = Conv2DTranspose(filters, 3)(conv2_3)
    conv1_4 = keras.layers.concatenate([up1_4, conv1_1, conv1_2, conv1_3])
    conv1_4 = add_2D_conv(conv1_4, filters, 3, separable=False)
    
    conv5_1 = add_2D_conv(pool4, filters*16, 3, separable=False)
    
    up4_2 = Conv2DTranspose(filters*8, 3)(conv5_1)
    conv4_2 = keras.layers.concatenate([up4_2, conv4_1])
    conv4_2 = add_2D_conv(conv4_2, filters*8, 3, separable=False)
    
    up3_3 = Conv2DTranspose(filters*4, 3)(conv4_2)
    conv3_3 = keras.layers.concatenate([up3_3, conv3_1, conv3_2])
    conv3_3 = add_2D_conv(conv3_3, filters*4, 3, separable=False)
    
    up2_4 = Conv2DTranspose(filters*2, 3)(conv3_3)
    conv2_4 = keras.layers.concatenate([up2_4, conv2_1, conv2_2, conv2_3])
    conv2_4 = add_2D_conv(conv2_4, filters*2, 3, separable=False)
    
    up1_5 = Conv2DTranspose(filters, 3)(conv2_4)
    conv1_5 = keras.layers.concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4])
    
    conv1_5 = add_2D_conv(unet, filters, 3, separable=False)
    
    split = conv1_5

    unet = add_2D_conv(conv1_5, filters, 3, separable=False)
    unet = add_2D_conv(unet, filters, 3, separable=False)

    out_binary_lst = []

    if binary_cutoffs:
        for d in binary_cutoffs:
            out_binary_lst.append(_add_binary_head(unet, d, 7))

    print (out_binary_lst)
    print ()

    unet = add_2D_conv(split, filters, 3, separable=False)
    unet = add_2D_conv(unet, filters, 3, separable=False)

    out_sscore = Conv2D(1, 7, activation ="sigmoid", data_format = "channels_last", 
            padding = "same", kernel_initializer = INIT, kernel_regularizer = REG, name = "out_sscore")(unet)
            
    print (out_sscore)
    
    model = Model(inputs = inputs_2D + inputs_seq, outputs = [out_sscore] + out_binary_lst)

    return model
    
"-------------------------------------------------------------------"

if __name__ == "__main__":

    modelfile = sys.argv[1]
    suffix = os.path.splitext(modelfile)[0]
    #num_blocks = int(sys.argv[2])

    binary_cutoffs = [6, 8, 10]

    model = create_unet_plus(binary_cutoffs=binary_cutoffs)
    #plot_model(model, "unet.png")
    print(model.summary())
    model = wrap_model(model, binary_cutoffs)
    model.save(modelfile)


"-------------------------------------------------------------------"
