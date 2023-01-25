import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Input, Add, PReLU, Lambda
import numpy as np

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

