import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, Activation, Input, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers

img_size = 256
n_batch = 32
n_class = 2
epoch = 50
AUTOTUNE = tf.data.AUTOTUNE

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'data', 'dataset')
train_dir = os.path.join(data_dir, 'training_set')
validation_dir = os.path.join(data_dir, 'test_set')

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

regularizer = regularizers.l2(l2=1e-4)

def build_AlexNet():
    input_layer = Input((img_size, img_size, 3))
    x = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid')(input_layer)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=regularizer)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizer)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizer)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizer)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = Flatten()(x)

    x = Dense(units=4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(units=4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(units=n_class, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output)

    return model

AlexNet = build_AlexNet()

filename = 'checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(epoch, n_batch)
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
earlystopping = EarlyStopping(monitor='val_accuracy', patience=10)

optimizer = keras.optimizers.Adam(lr = 0.0002)
losses = keras.losses.SparseCategoricalCrossentropy()

AlexNet.compile(
    loss = losses,
    optimizers = optimizer,
    metrics = ["accuracy"]
)

AlexNet.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch,
    verbose=1,
    callbacks=[checkpoint, earlystopping]
)