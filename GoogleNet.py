import tensorflow as tf
from keras.layers import Dropout, AveragePooling2D, Dense, Conv2D, MaxPooling2D, Activation, Concatenate
from keras.layers import GlobalAveragePooling2D, Flatten, Input
from keras.models import Model

def inception_block(x_in, x1_f, x3r_f, x3_f, x5r_f, x5_f, po):
    x1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_in)
    x1 = Conv2D(filters=po, kernel_size=(1, 1), padding="SAME")(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(filters=x5r_f, kernel_size=(1, 1), padding="same")(x_in)
    x2 = Conv2D(filters=x5_f, kernel_size=(5, 5), padding="same")(x2)
    x2 = Activation('relu')(x2)

    x3 = Conv2D(filters=x3r_f, kernel_size=(1, 1), padding="same")(x_in)
    x3 = Conv2D(filters=x3_f, kernel_size=(3, 3), padding="same")(x3)
    x3 = Activation('relu')(x3)

    x4 = Conv2D(filters=x1_f, kernel_size=(1, 1), padding="same")(x_in)
    x4 = Activation('relu')(x4)

    out = Concatenate()([x1, x2, x3, x4])
    return out


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