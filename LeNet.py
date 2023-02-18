import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.models import Sequential
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import os

img_size = 128
batch_size = 128
epoch = 50
AUTOTUNE = tf.data.AUTOTUNE
L1 = keras.regularizers.l2(2e-4)
n_class = 2
LR = 1e-4
L1 = keras.regularizers.l2(2e-4)

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

    model.add(Conv2D(filters=filter_1, kernel_size=5, strides=1, activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=L1, input_shape=(img_size, img_size, 3)))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=filter_2, kernel_size=5, strides=1, activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=L1))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=filter_3, kernel_size=5, strides=1, activation='relu', kernel_initializer='he_normal',
                     kernel_regularizer=L1))

    model.add(Flatten())
    model.add(Dense(units=84, activation='relu', kernel_regularizer=L1))

    model.add(Dense(units=n_class, activation='softmax'))

    return model

model = lenet(6, 16, 120)

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

losses = SparseCategoricalCrossentropy()
optimizers = Adam(learning_rate=learning_rate_fn)

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

model.evaluate(val_ds, verbose=1)

model.save('./model/lenet5', include_optimizer=False)

for image in val_ds.take(1):
    pred = model.predict(tf.expand_dims(image[0][0], 0)).argmax()
    image = np.array(image[0][0]).astype(np.uint8)
    plt.imshow(image)
    plt.show()
    if pred == 0:
        print("cat")
    else:
        print("dog")
