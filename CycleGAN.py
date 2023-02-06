import functools
from glob import glob
import tqdm
import matplotlib.pyplot as plt
import IPython.display as display
import PIL
import imageio
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import sys
# from experiment_utils import *
import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, BatchNormalization, Activation, add, Input, Conv2DTranspose, ZeroPadding2D, LeakyReLU
from tensorflow_addons.layers import InstanceNormalization
from keras.models import Model
from PIL import Image
import os, random, keras

input_shape = (128, 128, 3)
hidden_layers = 3
n_batch = 1
LR = 2e-4
BETA_1 = 0.5
epoch_decay = 100
epochs = 125
cycle_loss_weight = 10.0
identity_loss_weight = 0.0
gradient_penalty_weight = 10.0
gradient_penalty_mode = 'none'
autotune = tf.data.AUTOTUNE

cur_path = os.getcwd()
data_path = os.path.join(cur_path, 'data\\gan\\facades\\facades')

train_a_path = os.path.join(data_path, 'trainA')
train_b_path = os.path.join(data_path, 'trainB')
test_a_path = os.path.join(data_path, 'testA')
test_b_path = os.path.join(data_path, 'testB')

def get_image(path):
    dir = [file_name for file_name in os.listdir(path) if os.path.splitext(file_name)[-1] == '.jpg']
    image = np.array([np.array(Image.open(os.path.join(path, img))) for img in dir])
    print(f'image length : {len(image)}')
    return image

train_a = get_image(train_a_path)
train_b = get_image(train_b_path)
test_a = get_image(test_a_path)
test_b = get_image(test_b_path)

print(train_a.shape, train_b.shape, test_a.shape, test_b.shape)
print(train_a.max(), train_b.max(), test_a.max(), test_b.max())
print(train_a.min(), train_b.min(), test_a.min(), test_b.min())

ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
seed = random.seed(7777777)

def preprocessing(data):
    image = tf.cast(data, tf.uint8)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, [128, 128])
    image = (image-127.5) / 127.5
    return image

def custom_dataloader(data, mode):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.map(preprocessing, num_parallel_calls=autotune)
    if mode == 'test':
        dataset = dataset.repeat(n_batch * epochs)
    dataset = dataset.batch(n_batch)
    dataset = dataset.prefetch(autotune)

    return dataset

train_a = custom_dataloader(train_a, mode='train')
train_b = custom_dataloader(train_b, mode='train')
test_a = custom_dataloader(test_a, mode='test')
test_b = custom_dataloader(test_b, mode='test')

train_ds = tf.data.Dataset.zip((train_a, train_b))
test_ds = tf.data.Dataset.zip((test_a, test_b))

def residual_block(input_layer):

    x = input_layer
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-5)(x)

    return add([input_layer, x])

def InstanceNorm(input_layer, filter, kernel_size, stride):

    x = Conv2D(filters=filter, kernel_size=kernel_size, strides=stride, padding='same')(input_layer)
    x = InstanceNormalization(axis=1)(x)
    x = Activation('relu')(x)

    return x

def Transpose(input_layer, filter, kernel_size, stride, use_bias=False):

    x = Conv2DTranspose(filters=filter, kernel_size=kernel_size, strides=stride, padding='same',
                        use_bias=use_bias)(input_layer)
    x = InstanceNormalization(axis=1)(x)
    x = Activation('relu')(x)

    return x

def generator():

    input = Input(input_shape)

    x = InstanceNorm(input, 32, 7, 1)
    x = InstanceNorm(x, 64, 3, 2)
    x = InstanceNorm(x, 128, 3, 2)

    for _ in range(6):
        x = residual_block(x)

    x = Transpose(x, 64, 3, 2)
    x = Transpose(x, 32, 3, 2)

    x = Conv2D(filters=3, kernel_size=7, strides=1, padding='same')(x)
    output = Activation('tanh')(x)

    return Model(input, output)

def discriminator():

    input = Input(input_shape)

    x = ZeroPadding2D(padding=(1, 1))(input)
    x = Conv2D(filters=64, kernel_size=4, strides=2, padding='valid')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    for i in range(1, hidden_layers + 1):
        x = Conv2D(filters=2 ** i * 64, kernel_size=4, strides=2, padding='valid')(x)
        x = InstanceNormalization(axis=1)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = ZeroPadding2D(padding=(1, 1))(x)

    output = Conv2D(filters=1, kernel_size=4, strides=1, activation='sigmoid')(x)

    return Model(input, output)

Gen_a_to_b = generator()
Gen_b_to_a = generator()

Disc_a = discriminator()
Disc_b = discriminator()

