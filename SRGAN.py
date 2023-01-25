import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Input, Add, PReLU, Lambda, LeakyReLU, Dense
import numpy as np
from keras.models import Model

hr_img_size = 100
lr_img_size = 25
batch_size = 16
epoch = 30
AUTOTUNE = tf.data.AUTOTUNE

train, valid = tfds.load("div2k/bicubic_x4", split = ['train', 'validation'], as_supervised=True)

def preprocessing(low_image, high_image):
    high_scaling = tf.cast(high_image / 255, tf.float32)
    high_crop = tf.image.random_crop(high_scaling, size=(hr_img_size, hr_img_size, 3))
    low_crop = tf.image.resize(high_crop, (lr_img_size, lr_img_size), method="bicubic")

    return low_crop, high_crop

train_ds = train.map(preprocessing).shuffle(buffer_size=10).repeat().batch(batch_size)
val_ds = valid.map(preprocessing).repeat().batch(batch_size)

def gen_res_block(input_layer):

    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1,2])(x)
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    result = Add()([input_layer, x])

    return result

def gen_upsample(input_layer):

    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(input_layer)
    x = Lambda(lambda x : tf.nn.depth_to_space(x, 2))(x)
    result = PReLU(shared_axes=[1, 2])(x)

    return result

def disc_conv_block(input_layer, n_filter=64, first_step=False):

    if first_step:
        x = Conv2D(filters=n_filter, kernel_size=3, strides=2, padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
    else:
        x = input_layer
        for stride in range(2):
            x = Conv2D(filters=n_filter, kernel_size=3, strides=stride, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)

    return x

def build_generator():

    inputs = Input((None, None, 3))

    x = Conv2D(filters=64, kernel_size=9, strides=1, padding='same')(inputs)
    x = residual = PReLU(shared_axes=[1, 2])(x)

    for _ in range(5):
        x = gen_res_block(x)

    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([residual, x])

    for _ in range(2):
        x = gen_upsample(x)

    outputs = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(x)

    model = Model(inputs, outputs)

    return model

def build_discriminator():

    inputs = Input((None, None, 3))

    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(inputs)
    x = LeakyReLU()(x)

    x = disc_conv_block(x, first_step=True)

    for num in [128, 256, 512]:
        x = disc_conv_block(x, n_filter=num)

    x = Dense(1024)(x)
    x = LeakyReLU()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    return model



