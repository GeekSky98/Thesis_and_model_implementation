import os
from glob import glob
import time
import random

import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
from PIL import Image
import imageio
import numpy as np

import tensorflow as tf
from keras.layers import Conv2D, Dense, Conv2DTranspose, BatchNormalization, ReLU, Dropout, Flatten, Reshape, LeakyReLU
from keras.models import Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

n_batch = 64
epochs = 9000
depth = 100
img_size = 100
lr = 1e-4
autotune = tf.data.AUTOTUNE
noise = tf.random.normal([1, 100])
critic_weight = 0.01
n_critic = 5
beta = 0.5
L1 = keras.regularizers.l2(2e-4)

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'data', 'cars_images')

wgan_path = os.path.join(cur_dir, 'wgan')
checkpoint_dir = os.path.join(wgan_path, 'checkpoint')
output_dir = os.path.join(wgan_path, 'output')
epoch_image_dir = os.path.join(output_dir, 'epoch_image')
train_summary_dir = os.path.join(output_dir, 'train_summary')

seed = random.seed(7777777)

data_path = glob(os.path.join(data_dir, '*.jpg'))

def preprocessing(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    image = (image-127.5) / 127.5
    return image

def dataloader(paths):
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.shuffle(len(paths))
    dataset = dataset.map(preprocessing)
    dataset = dataset.batch(10 * n_batch)
    dataset = dataset.prefetch(autotune)
    return dataset

train_ds = dataloader(data_path)

def build_generator():
    model = Sequential()

    model.add(Dense(units=(25*25*128), use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((25,25,128)))

    model.add(Conv2DTranspose(filters=128, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=L1, bias_initializer='zeros', bias_regularizer=L1))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=L1, bias_initializer='zeros', bias_regularizer=L1))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding='same', activation='tanh',
                              kernel_initializer='he_normal', kernel_regularizer=L1, bias_initializer='zeros'))

    return model

def build_critic():
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=5, strides=2, padding='same', input_shape=[100, 100, 3]))
    model.add(ReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=128, kernel_size=5, strides=2, padding='same'))
    model.add(ReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(units=1))

    return model

def get_critic_loss(real_value, fake_value):
    real_loss = -tf.reduce_mean(real_value)
    fake_loss = tf.reduce_mean(fake_value)
    return real_loss, fake_loss

def get_gen_loss(fake_value):
    fake_loss = -tf.reduce_mean(fake_value)
    return fake_loss

generator_optimzier = Adam(learning_rate=lr)
critic_optimizer = Adam(learning_rate=lr)

generator = build_generator()
critic = build_critic()

def summary(name_data_dict,
            step=None,
            types=['mean', 'std', 'max', 'min', 'sparsity', 'histogram', 'image'],
            historgram_buckets=None,
            name='summary'):
    def _summary(name, data):
        if data.shape == ():
            tf.summary.scalar(name, data, step=step)
        else:
            if 'mean' in types:
                tf.summary.scalar(name + '-mean', tf.math.reduce_mean(data), step=step)
            if 'std' in types:
                tf.summary.scalar(name + '-std', tf.math.reduce_std(data), step=step)
            if 'max' in types:
                tf.summary.scalar(name + '-max', tf.math.reduce_max(data), step=step)
            if 'min' in types:
                tf.summary.scalar(name + '-min', tf.math.reduce_min(data), step=step)
            if 'sparsity' in types:
                tf.summary.scalar(name + '-sparsity', tf.math.zero_fraction(data), step=step)
            if 'histogram' in types:
                tf.summary.histogram(name, data, step=step, buckets=historgram_buckets)
            if 'image' in types:
                tf.summary.image(name, data, step=step)

    with tf.name_scope(name):
        for name, data in name_data_dict.items():
            _summary(name, data)

train_summary_writer = tf.summary.create_file_writer(os.path.join(train_summary_dir, 'summaries', 'train'))
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimzier,
                                 discriminator_optimizer=critic_optimizer,
                                 generator=generator,
                                 discriminator=critic)

def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    dtype = dtype if dtype else images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)

def generate_and_save_images(model, epoch, noise):
    plt.figure(figsize=(15, 10))

    for i in range(4):
        images = model(noise, training=False)

        image = images[0, :, :, :]
        image = np.reshape(image, [100, 100, 3])

        image = to_range(image, 0, 255, dtype=np.uint8)

        plt.subplot(1, 4, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title("Randomly Generated Images")

        print(image)

    plt.tight_layout()
    plt.savefig(os.path.join(epoch_image_dir, 'image_at_epoch_{:02d}.png'.format(epoch)))
    plt.show()

def train_generator(noise):
    with tf.GradientTape() as generator_tape:
        fake_image = generator(noise, training=True)

        fake_value = critic(fake_image, training=True)

        gen_loss = get_gen_loss(fake_value)

    gen_gradient = generator_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimzier.apply_gradients(zip(gen_gradient, generator.trainable_variables))

    return{'generator loss' : gen_loss}

def train_critic(noise, real_img):
    with tf.GradientTape() as critic_tape:
        fake_image = generator(noise, training=True)

        real_value = critic(real_img, training=True)
        fake_value = critic(fake_image, training=True)

        real_loss, fake_loss = get_critic_loss(real_value, fake_value)
        critic_loss = (real_loss + fake_loss)

    critic_gradient = critic_tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_gradient, critic.trainable_variables))
    for w in critic.trainable_variables:
        w.assign(tf.clip_by_value(w, -critic_weight, critic_weight))

    return {'critic loss' : real_loss + fake_loss}

with train_summary_writer.as_default():
    with tf.summary.record_if(True):
        for epoch in range(epochs):
            start_time = time.time()

            for batch in train_ds:
                critic_loss_dict = train_critic(noise, batch)

            summary(critic_loss_dict, step=critic_optimizer.iterations, name='critic_losses')

            if critic_optimizer.iterations.numpy() % n_critic == 0:
                gen_loss_dict = train_generator(noise)
                summary(gen_loss_dict, step=generator_optimzier.iterations, name='generator_losses')

            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
                display.clear_output(wait=True)
                generate_and_save_images(generator, epoch + 1, noise)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start_time))