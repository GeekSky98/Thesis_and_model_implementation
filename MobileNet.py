import tensorflow as tf
import numpy as np
from keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, Activation, Concatenate, DepthwiseConv2D
from keras.layers import Flatten, Input, LayerNormalization, AveragePooling2D, BatchNormalization
from keras.models import Model
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import os

img_size = 256
batch_size = 32
epoch = 50
AUTOTUNE = tf.data.AUTOTUNE
L1 = keras.regularizers.l2(2e-4)

def conv_block(input_layer, stride, channel, padd):

    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=channel, kernel_size=(1, 1), strides=stride, padding=padd)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def build_mobilenet():

    input = Input((None, None, 3))

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for i in range(6):
        temp_stride=0 if i % 2 == 0 else temp_stride=1
        x = conv_block(x, temp_stride, )