import tensorflow as tf
from keras.layers import Dropout, AveragePooling2D, Dense, Conv2D, MaxPooling2D, Activation, Concatenate
from keras.layers import GlobalAveragePooling2D, Flatten, Input, LayerNormalization, AveragePooling2D
from keras.models import Model
from tensorflow import keras

L1 = keras.regularizers.l2(0.0002)

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

    return Concatenate(axis=-1)([x1, x2, x3, x4])

def build_googlenet():

    input = Input((224, 224, 3))

    x_1 = Conv2D(filters=64, kernel_size=(7, 7), strides=2, activation='relu', padding='same',
                 kernel_initializer='he_normal', kernel_regularizer=L1)(input)
    x_1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x_1)
    x_1 = LayerNormalization()(x_1)

    x_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, activation='relu', padding="valid",
                 kernel_initializer='he_normal', kernel_regularizer=L1)(x_1)
    x_2 = Conv2D(filters=192, kernel_size=(3, 3), strides=1, activation='relu', padding="same",
                 kernel_initializer='he_normal', kernel_regularizer=L1)(x_2)
    x_2 = LayerNormalization()(x_2)

    x_3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x_2)

    inception_1 = inception_block(x_3, 64, 96, 128, 16, 32, 32)

    inception_2 = inception_block(inception_1, 128, 128, 192, 32, 96, 64)

    x_4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(inception_2)

    inception_3 = inception_block(x_4, 192, 96, 208, 16, 48, 64)

    inception_4 = inception_block(inception_3, 160, 112, 224, 24, 64, 64)

    aux_1 = AveragePooling2D(pool_size=(5, 5), strides=1, padding='valid')(inception_3)
    aux_1 = Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding='same', activation='relu',
                   kernel_initializer='he_normal', kernel_regularizer=L1)(aux_1)
    aux_1 = Flatten()(aux_1)
    aux_1 = Dense(units=1024, activation='relu', kernel_initializer='he_normal', kernel_regularizer=L1)(aux_1)
    aux_1 = Dropout(rate=0.7)(aux_1)
    aux_1 = Dense(units=1000, activation='relu', kernel_initializer='he_normal', kernel_regularizer=L1)(aux_1)
    aux_1 = Activation('softmax')(aux_1)

    inception_5 = inception_block(inception_4, 128, 128, 256, 24, 64, 64)

    inception_6 = inception_block(inception_5, 112, 144, 288, 32, 64, 64)

    inception_7 = inception_block(inception_6, 256, 160, 320, 32, 128, 128)

    aux_2 = AveragePooling2D(pool_size=(5, 5), strides=1, padding='valid')(inception_6)
    aux_2 = Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding='same', activation='relu',
                   kernel_initializer='he_normal', kernel_regularizer=L1)(aux_2)
    aux_2 = Flatten()(aux_2)
    aux_2 = Dense(units=1024, activation='relu', kernel_initializer='he_normal', kernel_regularizer=L1)(aux_2)
    aux_2 = Dropout(rate=0.7)(aux_2)
    aux_2 = Dense(units=1000, activation='relu', kernel_initializer='he_normal', kernel_regularizer=L1)(aux_2)
    aux_2 = Activation('softmax')(aux_2)

    x_5 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(inception_7)

    inception_8 = inception_block(x_5, 256, 160, 320, 32, 128, 128)

    inception_9 = inception_block(inception_8, 384, 192, 384, 48, 128, 128)

    x_6 = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid')(inception_9)

    x_fc = Flatten()(x_6)
    x_fc = Dropout(rate=0.4)(x_fc)
    x_fc = Dense(units=1000, kernel_initializer='he_normal', kernel_regularizer=L1)(x_fc)

    output = Activation('softmax')(x_fc)

    return Model(inputs=input, outputs=[aux_1, aux_2, output])

googlenet = build_googlenet()

googlenet.summary()