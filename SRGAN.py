import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Input, Add, PReLU, Lambda, LeakyReLU, Dense
from keras.losses import BinaryCrossentropy, MeanSquaredError
import tensorflow as tf
import numpy as np
from keras.models import Model
from tensorflow import keras
from keras.applications.vgg19 import VGG19, preprocess_input
import keras.applications.vgg19 as vgg
from tensorflow.keras import optimizers, metrics
from skimage.transform import resize

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

def gen_res_block(input_layer):

    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1,2])(x)
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    result = Add()([input_layer, x])

    return result

def gen_upsample(input_layer):

    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(input_layer)
    x = Lambda(lambda x : tf.nn.depth_to_space(x, 2))(x)
    result = PReLU(shared_axes=[1, 2])(x)

    return result

def disc_conv_block(input_layer, n_filter=64, first_step=False):

    if first_step:
        x = Conv2D(filters=n_filter, kernel_size=3, strides=2, padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
    else:
        x = input_layer
        for stride in range(1,3):
            x = Conv2D(filters=n_filter, kernel_size=3, strides=stride, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)

    return x

def build_generator():

    inputs = Input((None, None, 3))

    x = Conv2D(filters=64, kernel_size=9, strides=1, padding='same')(inputs)
    x = residual = PReLU(shared_axes=[1, 2])(x)

    for _ in range(5):
        x = gen_res_block(x)

    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([residual, x])

    for _ in range(2):
        x = gen_upsample(x)

    outputs = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(x)

    model = Model(inputs, outputs)

    return model

def build_discriminator():

    inputs = Input((None, None, 3))

    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(inputs)
    x = LeakyReLU()(x)

    x = disc_conv_block(x, first_step=True)

    for num in [128, 256, 512]:
        x = disc_conv_block(x, n_filter=num)

    x = Dense(1024)(x)
    x = LeakyReLU()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    return model

vgg_transfer = vgg.VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
vgg_transfer.summary()

vgg_model = Model(vgg_transfer.input, vgg_transfer.get_layer('block5_conv4').output)


generator = build_generator()
discriminator = build_discriminator()

def get_generator_loss(fake_output):
    return BinaryCrossentropy(tf.ones_like(fake_output), fake_output, from_logits=False)

def get_discriminator_loss(real_output, fake_output):
    return BinaryCrossentropy(tf.ones_like(real_output), real_output) + \
           BinaryCrossentropy(tf.zeros_like(fake_output), fake_output, from_logits=False)

def get_content_loss_vgg(real, fake):
    real, fake = vgg.preprocess_input(real), vgg.preprocess_input(fake)

    real_feature_map = vgg_model(real) / 12.75
    fake_feature_map = vgg_model(fake) / 12.75

    return MeanSquaredError(real_feature_map, fake_feature_map)

generator_optimizer = optimizers.Adam()
discriminator_optimizer = optimizers.Adam()

def gaaaan(low_img, high_real):
    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
        fake = generator(low_img, training=True)

        real_out = discriminator(high_real, training=True)
        fake_out = discriminator(fake, training=True)

        perceptual_loss = get_content_loss_vgg(high_real, fake) + 1e-3 * get_generator_loss(fake_out)
        discriminator_loss = get_discriminator_loss(real_out, fake_out)

    generator_gradient = generator_tape.gradient(perceptual_loss, generator.trainable_variables)
    discriminator_gradient = discriminator_tape(discriminator_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))

    return perceptual_loss, discriminator_loss

generator_losses = metrics.Mean()
discriminator_losses = metrics.Mean()

for epoch in range(1, 2):
    for i, (low_img, high_img) in enumerate(train):
        g_loss, d_loss = gaaaan(low_img, high_img)

        generator_losses.update_state(g_loss)
        discriminator_losses.update_state(d_loss)

        if (i + 1) % 10 == 0:
            print(
                f"EPOCH[{epoch}] - STEP[{i + 1}] \nGenerator_loss:{generator_losses.result():.4f} \nDiscriminator_loss:{discriminator_losses.result():.4f}",
                end="\n\n")

        if (i + 1) == 200:
            break

    generator_losses.reset_states()
    discriminator_losses.reset_states()

def result(image):
    image = np.array(image)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.float32)
    pred = generator.predict(image)
    pred = tf.clip_by_value(pred, 0, 255)
    pred = tf.cast(round(pred), tf.uint8)

    return tf.squeeze(np.array(pred), axis=0)

def contrast(image):
    hr_image = np.array(image)
    img_height, img_width = hr_image.shape[0], hr_image.shape[1]
    lr_image = resize(hr_image, (img_height//4, img_width//4, 3), 'bicubic')
    srgan_image = result(lr_image)
    bicubic_image = resize(lr_image, (img_height, img_width, 3), 'bicubic')

    plt.imshow(np.concatenate([bicubic_image, srgan_image, hr_image], axis=1))
    plt.show()