import keras.backend
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Activation, AveragePooling2D
from keras.layers import Concatenate, GlobalAveragePooling2D, Dense
from keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from skimage.transform import resize
from PIL import Image
import os

img_size = 256
batch_size = 32
epoch = 50
n_class = 2
L1 = keras.regularizers.l2(2e-4)
AUTOTUNE = tf.data.AUTOTUNE
repeat = {
    'DenseNet_112' : [6, 12, 24, 16],
    'DenseNet_169' : [6, 12, 32, 32],
    'DenseNet_201' : [6, 12, 48, 32],
    'DenseNet_264' : [6, 12, 64, 48]
}

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'data', 'dataset')
train_dir = os.path.join(data_dir, 'training_set')
validation_dir = os.path.join(data_dir, 'test_set')

train_ds = keras.preprocessing.image_dataset_from_directory(
    train_dir,
    shuffle = True,
    image_size = (img_size, img_size),
    batch_size = batch_size
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    validation_dir,
    shuffle = True,
    image_size = (img_size, img_size),
    batch_size = batch_size
)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

def Conv_block(input_layer, growth_rate):

    x = BatchNormalization()(input_layer)
    x = Activation('relu')(x)
    x = Conv2D(filters=growth_rate*4, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=growth_rate, kernel_size=3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=L1)(x)
    x = Dropout(rate=0.2)(x)

    return Concatenate(axis=-1)([input_layer, x])

def Dense_block(input_layer, repetiton, base_channel=32):
    x = input_layer
    for _ in range(repetiton):
        x = Conv_block(x, base_channel)

    return x

def Transition_layer(input_layer, reduce_ratio=.5):

    channel = int(keras.backend.int_shape(input_layer)[-1] * reduce_ratio)

    x = BatchNormalization()(input_layer)
    x = Activation('relu')(x)
    x = Conv2D(filters=channel, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=L1)(x)
    x = Dropout(rate=0.2)(x)
    x = AveragePooling2D(pool_size=2, strides=2)(x)

    return x

def build_DenseNet(version="DenseNet_112"):

    input = Input((img_size, img_size, 3))

    x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=L1)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = Dense_block(x, repeat[version][0], base_channel=32)
    x = Transition_layer(x, reduce_ratio=.5)
    x = Dense_block(x, repeat[version][1], base_channel=32)
    x = Transition_layer(x, reduce_ratio=.5)
    x = Dense_block(x, repeat[version][2], base_channel=32)
    x = Transition_layer(x, reduce_ratio=.5)
    x = Dense_block(x, repeat[version][3], base_channel=32)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=0.2)(x)

    output = Dense(n_class, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=L1)(x)

    return Model(input, output)

densenet = build_DenseNet()

densenet.summary()

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

losses = SparseCategoricalCrossentropy()
optimizers = SGD(
    learning_rate=learning_rate_fn,
    momentum=0.9
)

densenet.compile(
    loss=losses,
    optimizer=optimizers,
    metrics=['accuracy']
)

filename = 'checkpoint2-epoch-{}-batch-{}-trial-001.h5'.format(epoch, batch_size)
checkpoint = ModelCheckpoint(filename, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
earlystopping = EarlyStopping(monitor='val_accuracy', patience=20)

densenet.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch,
    verbose=1,
    callbacks=[checkpoint, earlystopping]
)
