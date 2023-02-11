import tensorflow as tf
from keras.layers import MaxPool2D, Conv2D, Flatten, Dense, AveragePooling2D
from keras.models import Sequential
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
import os

img_size = 256
batch_size = 32
epoch = 50
AUTOTUNE = tf.data.AUTOTUNE
L1 = keras.regularizers.l2(2e-4)
n_class = 2
LR = 1e-4

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


def lenet(filter_1, filter_2, filter_3):

    model = Sequential()

    model.add(Conv2D(filters=filter_1, kernel_size=5, strides=1, activation='relu',
                     input_shape=(img_size, img_size, 3)))
    model.add(AveragePooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=filter_2, kernel_size=5, strides=1, activation='relu'))
    model.add(AveragePooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=filter_3, kernel_size=5, strides=1, activation='relu'))

    model.add(Flatten())
    model.add(Dense(units=84, activation='tanh'))
    model.add(Dense(units=n_class, activation='softmax'))

    return model

model = lenet(6, 16, 120)

losses = SparseCategoricalCrossentropy()
optimizers = Adam(learning_rate=LR)

model.compile(
    loss = losses,
    optimizer = optimizers,
    metrics = ['accuracy']
)

filename = 'checkpoint2-epoch-{}-batch-{}-trial-001.h5'.format(epoch, batch_size)
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
earlystopping = EarlyStopping(monitor='val_loss', patience=20)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch,
    verbose=1,
    callbacks=[checkpoint, earlystopping]
)