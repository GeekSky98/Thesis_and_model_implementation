import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, Activation, Input, MaxPool2D, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model
from tensorflow.keras.applications import *
import os
from PIL import Image
from keras.callbacks import ModelCheckpoint, EarlyStopping

layer_idx = [3, 4, 6, 3]
channel_idx = [64, 128, 256, 512]
n_class = 2
img_size = 320
n_batch = 32
epoch = 50
AUTOTUNE = tf.data.AUTOTUNE

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'data', 'dataset')
train_dir = os.path.join(data_dir, 'training_set')
validation_dir = os.path.join(data_dir, 'test_set')

train_dir_cat = os.path.join(train_dir, 'cats')
val_dir_cat = os.path.join(validation_dir, 'cats')
train_dir_dog = os.path.join(train_dir, 'dogs')
val_dir_dog = os.path.join(validation_dir, 'dogs')
dir_list = [train_dir_cat, val_dir_cat, train_dir_dog, val_dir_dog]

for dir in dir_list:
    temp_list = [fname for fname in os.listdir(dir)]
    for file in temp_list:
        temp = Image.open(os.path.join(dir, file))
        if os.path.splitext(file)[-1] != '.jpeg' and temp.mode != 'RGB':
            os.remove(os.path.join(dir, file))
            print(file + " delete")
        else:
            continue

for files in os.listdir(train_dir_dog)[100:140]:
    file = os.path.join(train_dir_dog, files)
    print(np.array(Image.open(file)).shape)

train_ds = keras.preprocessing.image_dataset_from_directory(
    train_dir,
    shuffle = True,
    image_size = (img_size, img_size),
    batch_size = n_batch
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    validation_dir,
    shuffle = True,
    image_size = (img_size, img_size),
    batch_size = n_batch
)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

def ResNet50_residual_block(input_layer, n_layer, channel, stride):

    x = input_layer
    residual_value = x
    residual_value = Conv2D(filters=(channel * 4), kernel_size=(1, 1), padding='same', strides=stride)(residual_value)
    residual_value = BatchNormalization()(residual_value)

    for n_layer in range(n_layer):
        x = Conv2D(filters=channel, kernel_size=(1, 1), strides=stride, kernel_initializer='he_normal',
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=channel, kernel_size=(3, 3), strides=1, kernel_initializer='he_normal', padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=(channel * 4), kernel_size=(1, 1), strides=1, kernel_initializer='he_normal',
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, residual_value])
        if stride != 1:
            stride = 1

        x = Activation('relu')(x)
        residual_value = x

    return x

def build_ResNet50(img_size):

    input_layer = Input((img_size, img_size, 3))
    output = Conv2D(filters=channel_idx[0], kernel_size=(1, 1), strides=2, kernel_initializer='he_normal',
                    padding='same')(input_layer)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = MaxPool2D(pool_size=(3, 3), strides=2)(output)

    prev_channel = 64
    for i, (n_layer, channel) in enumerate(zip(layer_idx, channel_idx)):
        stride = 1 if channel == prev_channel else 2
        output = ResNet50_residual_block(output, n_layer=n_layer, channel=channel, stride=stride)
        prev_channel = channel

    output = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(output)
    output = Flatten()(output)
    output = Dense(n_class, activation='softmax')(output)

    model = Model(inputs=input_layer, outputs=output)

    return model

resnet_50 = build_ResNet50(img_size)

resnet_50.summary()

filename = 'checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(epoch, n_batch)
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
earlystopping = EarlyStopping(monitor='val_accuracy', patience=10)

optimizer = keras.optimizers.Adam(
    learning_rate=0.0002,
    beta_1=0.93,
    beta_2=0.999,
    epsilon=2e-08,
    amsgrad=True
)

losses = keras.losses.SparseCategoricalCrossentropy()

resnet_50.compile(
    loss = losses,
    optimizer = optimizer,
    metrics = ["accuracy"]
)

resnet_50.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch,
    verbose=1,
    callbacks=[checkpoint, earlystopping]
)