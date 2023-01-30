import tensorflow as tf
from keras.layers import Dropout, AveragePooling2D, Dense, Conv2D, MaxPooling2D, Activation, Concatenate
from keras.layers import GlobalAveragePooling2D, Flatten, Input, LayerNormalization
from keras.models import Model

def inception_block(input_layer, x1_conv, x2_conv, x2_conv2, x3_conv, x3_conv2, x4_conv):
    x1 = Conv2D(filters=x1_conv, kernel_size=(1, 1), padding="SAME")(input_layer)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(filters=x2_conv, kernel_size=(1, 1), padding="same")(input_layer)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(filters=x2_conv2, kernel_size=(3, 3), padding="same")(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(filters=x3_conv, kernel_size=(1, 1), padding="same")(input_layer)
    x3 = Activation('relu')(x3)
    x3 = Conv2D(filters=x3_conv2, kernel_size=(5, 5), padding="same")(x3)
    x3 = Activation('relu')(x3)

    x4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_layer)
    x4 = Conv2D(filters=x4_conv, kernel_size=(1, 1), padding="same")(x4)
    x4 = Activation('relu')(x4)

    return Concatenate([x1, x2, x3, x4], axis=-1)

def build_googlenet():

    inputs = Input((None, None, 3))

    x_1 = Conv2D(filters=64, kernel_size=(7, 7), strides=2, activation='relu')(inputs)
    x_1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x_1)
    x_1 = LayerNormalization()(x_1)

    x_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x_1)
    x_2 = Conv2D(filters=192, kernel_size=(3, 3), strides=2, padding='same')(x_2)

inputs = Input(shape=(224, 224, 3))
x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same")(inputs)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
x = tf.keras.layers.LayerNormalization()(x)

x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
x = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)

x = tf.keras.layers.LayerNormalization()(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

x = inception_block(x, 64, 96, 128, 16, 32, 32)
x = inception_block(x, 128, 128, 192, 32, 96, 64)

x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
x = inception_block(x, 192, 96, 208, 16, 48, 64)

ax1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
ax1 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")(ax1)
ax1 = Flatten()(ax1)
ax1 = Dense(1024, activation="relu")(ax1)
ax1 = Dropout(0.7)(ax1)
ax1 = Dense(1000, activation="softmax")(ax1)

x = inception_block(x, 160, 112, 224, 24, 64, 64)
x = inception_block(x, 128, 128, 256, 24, 64, 64)

x = inception_block(x, 112, 114, 288, 32, 64, 64)

ax2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(x)
ax2 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")(ax2)
ax2 = Flatten()(ax2)
ax2 = Dense(1024, activation="relu")(ax2)
ax2 = Dropout(0.7)(ax2)
ax2 = Dense(1000, activation="softmax")(ax2)

x = inception_block(x, 256, 160, 320, 32, 128, 128)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

x = inception_block(x, 256, 160, 320, 32, 128, 128)
x = inception_block(x, 384, 192, 384, 48, 128, 128)

x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)

outputs = Dense(1000, activation="softmax")(x)
model = Model(inputs, [outputs, ax1, ax2])
model.summary()