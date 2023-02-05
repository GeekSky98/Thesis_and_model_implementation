import tensorflow as tf
import numpy as np
from keras.layers import Dropout, Dense, Conv2D, Activation, DepthwiseConv2D
from keras.layers import Input, BatchNormalization, GlobalAveragePooling2D
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
n_class = 2
channels = [64, 128, 128, 256, 256, 512]

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

def conv_block(input_layer, stride, channel):

    x = DepthwiseConv2D(kernel_size=(3, 3), strides=stride, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=channel, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=L1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def build_mobilenet():

    input = Input((img_size, img_size, 3))

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=L1)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    num_st = 0
    for channel in channels:
        num_st += 1
        temp_stride=2 if num_st % 2 == 0 else 1
        x = conv_block(input_layer=x, stride=temp_stride, channel=channel)

    for _ in range(5):
        x = conv_block(input_layer=x, stride=1, channel=512)

    for i in range(2):
        stride=2 if i==0 else 1
        x = conv_block(input_layer=x, stride=stride, channel=1024)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)

    x = Dense(units=n_class, kernel_initializer='he_normal', kernel_regularizer=L1)(x)
    output = Activation('softmax')(x)

    return Model(input, output)


mobilenet = build_mobilenet()
mobilenet.summary()


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

mobilenet.compile(
    loss=losses,
    optimizer=optimizers,
    metrics=['accuracy']
)

filename = 'checkpoint2-epoch-{}-batch-{}-trial-001.h5'.format(epoch, batch_size)
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
earlystopping = EarlyStopping(monitor='val_loss', patience=20)

mobilenet.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch,
    verbose=1,
    callbacks=[checkpoint, earlystopping]
)