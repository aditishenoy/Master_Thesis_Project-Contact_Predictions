from __future__ import division
import os
from collections import namedtuple


import keras
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dropout, BacthNormalization
from keras.layers import Conv2D, Conv1D, MaxPooling2D, Upsampling2D
from keras.layers.core import Lambda
from keras.layers.advanced_activations import ELU

DROPOUT = 0.1
ACTIVATION = ELU
INIT = "he_normal"
REG = None

reg_strength = float(10**-12)
REG = l2(reg_strength)


