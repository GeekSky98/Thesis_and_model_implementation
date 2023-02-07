import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Reshape, BatchNormalization, ReLU, Conv2DTranspose, Conv2D, Dropout, Flatten
from keras.models import Sequential
from tensorflow.keras import Sequential
from keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import os, random, glob, time

n_batch = 64
epochs = 6000
depth = 300
img_size = 100
lr = 1e-4
autotune = tf.data.AUTOTUNE

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'data', 'cars_images')

dcgan_path = os.path.join(cur_dir, 'dcgan')
checkpoint_dir = os.path.join(dcgan_path, 'checkpoint')
output_dir = os.path.join(dcgan_path, 'output')
epoch_image_dir = os.path.join(output_dir, 'epoch_image')
train_summary_dir = os.path.join(output_dir, 'train_summary')

seed = random.seed(7777777)

data_path = glob.glob(os.path.join(data_dir, '*.jpg'))

def preprocessing(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    image = image / 255.0
    return image

def dataloader(paths):
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.shuffle(10 * n_batch)
    dataset = dataset.map(preprocessing)
    dataset = dataset.batch(n_batch)
    dataset = dataset.prefetch(autotune)
    return dataset

train_ds = dataloader(data_path)

def build_generator():

    model = Sequential()

    model.add(Dense(units=25*25*128, use_bias=False, input_shape=(depth,)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Reshape((25,25,128)))

    model.add(Conv2DTranspose(filters=128, kernel_size=5, strides=1, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding='same',activation='sigmoid', use_bias=False))

    return model

def build_discriminator():

    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=5, strides=2, padding='same', input_shape=(100, 100, 3)))
    model.add(ReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=128, kernel_size=5, strides=2, padding='same'))
    model.add(ReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model

binaryCE = BinaryCrossentropy()

def get_gen_loss(fake):
    return binaryCE(tf.ones_like(fake), fake)

def get_disc_loss(real, fake):

    real_loss = binaryCE(tf.ones_like(real), real)
    fake_loss = binaryCE(tf.zeros_like(fake), fake)
    losses = real_loss+fake_loss

    return losses

gen_optimizer = Adam(learning_rate=lr)
disc_optimizer = Adam(learning_rate=lr)
generator = build_generator()
discriminator = build_discriminator()

def summary(name_data_dict,
            step=None,
            types=['mean', 'std', 'max', 'min', 'sparsity', 'histogram'],
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

    with tf.name_scope(name):
        for name, data in name_data_dict.items():
            _summary(name, data)

train_summary_writer = tf.summary.create_file_writer(train_summary_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                 discriminator_optimizer=disc_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


def generate_and_save_images(model, epoch):
    plt.figure(figsize=(15, 10))

    for i in range(4):
        noise = tf.random.normal([1, depth])
        images = model(noise, training=False)

        image = images[0, :, :, :]
        image = np.reshape(image, [100, 100, 3])

        plt.subplot(1, 4, i + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title("Randomly Generated Images")

    plt.tight_layout()
    plt.savefig(os.path.join(epoch_image_dir, 'image_at_epoch_{:05d}.png'.format(epoch)))
    plt.show()

def train_step(image):
    noise = tf.random.normal([n_batch, depth])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_image = generator(noise, training=True)

        real_output = discriminator(image, training=True)
        fake_output = discriminator(fake_image, training=True)

        gen_loss = get_gen_loss(fake_output)
        disc_loss = get_disc_loss(real_output, fake_output)

    generator_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))

    return {'Generator loss': gen_loss,
            'Discriminator loss': disc_loss}

with train_summary_writer.as_default():
    with tf.summary.record_if(True):
        for epoch in range(epochs):
            start_time = time.time()

            for batch in train_ds:
                loss_dict = train_step(batch)
            summary(loss_dict, step=gen_optimizer.iterations, name='losses')
            if (epoch+1) % 15 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
                generate_and_save_images(generator, epoch + 1)
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start_time))

#finish
#finish
#finish
#finish
#finish
#finish
