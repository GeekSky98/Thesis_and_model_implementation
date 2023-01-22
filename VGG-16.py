import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense
from keras.models import Model
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image
from skimage.transform import resize

img_size = 256
n_batch = 32
n_class = 2
epoch = 30
n_conv = [2, 2, 3, 3, 3]
filter = [64, 128, 256, 512, 512]
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

for image, label in train_ds.take(1):
    plt.imshow(image[0].numpy().astype('uint8'))
    print(label[0].numpy())
    plt.show()

def conv_pool_block(input_layer):
    x = input_layer
    for idx in range(len(n_conv)):
        num_conv = n_conv[idx]
        for _ in range(num_conv):
            x = Conv2D(filters=filter[idx],
                       kernel_size=3,
                       activation="relu",
                       padding="same",
                       kernel_initializer="he_normal"
                       )(x)
        x = MaxPooling2D(pool_size=2, strides=2)(x)
    return x

def build_VGG16():
    input = Input(shape=(img_size, img_size, 3))

    x = conv_pool_block(input)

    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)
    x = Dense(4096, activation="relu")(x)
    output = Dense(n_class, activation="softmax")(x)

    model = Model(input, output)

    return model

model = build_VGG16()

model.summary()

optimizer = keras.optimizers.Adam()
losses = keras.losses.SparseCategoricalCrossentropy()

filename = 'checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(epoch, n_batch)
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
earlystopping = EarlyStopping(monitor='val_accuracy', patience=10)

model.compile(
    loss=losses,
    optimizer=optimizer,
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch,
    callbacks=[checkpoint, earlystopping],
    verbose=1
)

def result(image):
    temp = np.array(image)
    temp = resize(temp, (img_size, img_size, 3))
    plt.imshow(temp)
    plt.show()
    num = model.predict(np.expand_dims(temp, 0)).argmax()
    print('cat') if num == 0 else print('dog')

sample = Image.open('./data/sample.jpg')
result(sample)