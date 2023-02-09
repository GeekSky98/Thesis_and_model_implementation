import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Concatenate, Dense, Embedding, Flatten, Input, Reshape, Dropout
from keras.layers.advanced_activations import LeakyReLU, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.preprocessing import OrdinalEncoder
from glob import glob
import os, time

N_BATCH = 64
EPOCHS = 4000
LATENT_DEPTH = 100
IMAGE_SIZE = 100
LR = 1e-4
N_CLASS = 20
autotune = tf.data.AUTOTUNE

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'data', 'cars_images', '')

cgan_path = os.path.join(cur_dir, 'cgan')
checkpoint_dir = os.path.join(cgan_path, 'checkpoint')
output_dir = os.path.join(cgan_path, 'output')
epoch_image_dir = os.path.join(output_dir, 'epoch_image')
train_summary_dir = os.path.join(output_dir, 'train_summary')


data_path = glob(os.path.join(data_dir, '*.jpg'))

images_name = [i.split(data_dir)[:][1] for i in data_path]
car_name = [i.split('_')[0] for i in images_name]
len(np.unique(car_name))    # class : 20
car_array = np.reshape(car_name, (-1, 1))
encoder = OrdinalEncoder()
encoder.fit(car_array)
label_encoded = encoder.transform(car_array)

def preprocessing(path, labels):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = (image-127.5) / 127.5
    return image, labels

def dataloader(paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.shuffle(10*N_BATCH)
    dataset = dataset.map(preprocessing)
    dataset = dataset.batch(N_BATCH)
    dataset = dataset.prefetch(autotune)
    return dataset

train_ds = dataloader(data_path, label_encoded)

def build_generator():

    label = Input(shape=(1,))
    label_layer = Embedding(N_CLASS, 50)(label)
    label_layer = Dense(25 * 25)(label_layer)
    label_layer = Reshape(target_shape=(25, 25, 1))(label_layer)

    noise = Input(shape=(LATENT_DEPTH,))
    noise_layer = Dense(units=25 * 25 * 128)(noise)
    noise_layer = ReLU()(noise_layer)
    noise_layer = Reshape(target_shape=(25, 25, 128))(noise_layer)

    input = Concatenate()([noise_layer, label_layer])

    x = Conv2DTranspose(filters=128, kernel_size=5, strides=1, padding='same', use_bias=False)(input)
    x = ReLU()(x)

    x = Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False)(x)
    x = ReLU()(x)

    output = Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding='same', activation='tanh', use_bias=False)(x)

    return Model([noise, label], output)

def build_discriminator():
    input_label = Input(shape=(1,))
    input_label_embedding = Embedding(20, 50)(input_label)
    input_label_embedding = Dense(units=IMAGE_SIZE * IMAGE_SIZE)(input_label_embedding)
    input_label_embedding = Reshape(target_shape=(IMAGE_SIZE, IMAGE_SIZE, 1))(input_label_embedding)

    input_image = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    input = Concatenate()([input_image, input_label_embedding])

    x = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(input)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)

    output = Dense(units=1)(x)

    return Model([input_image, input_label], output)

generator = build_generator()
discriminator = build_discriminator()

gen_optimizer = Adam(learning_rate=LR)
disc_optimizer = Adam(learning_rate=LR)

losses = BinaryCrossentropy(from_logits=True)

def get_gen_loss(fake):
    return losses(tf.ones_like(fake), fake)

def get_disc_loss(real, fake):
    real_loss = losses(tf.ones_like(real), real)
    fake_loss = losses(tf.zeros_like(fake), fake)
    loss = real_loss + fake_loss
    return loss

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                 discriminator_optimizer=disc_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def to_range(images, min_value=0, max_value=255):
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(np.uint8)

def generate_and_save_images(model, label, epoch):
    plt.figure(figsize=(15, 10))

    for i in range(4):
        noise = tf.random.normal([1, LATENT_DEPTH])
        images = model([noise, label], training=False)
        image = images[0, :, :, :]
        image = np.reshape(image, [100, 100, 3])
        image = to_range(image)

        plt.subplot(1, 4, i + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title("Randomly Generated Images")

    plt.tight_layout()
    plt.savefig(os.path.join(epoch_image_dir, 'image_at_epoch_{:02d}.png'.format(epoch)))
    plt.show()
    print(label)

def train_step(image, label):
    noise_image = tf.random.normal([N_BATCH, LATENT_DEPTH])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_image = generator([noise_image, label], training=True)

        real_value = discriminator([image, label], training=True)
        fake_value = discriminator([fake_image, label], training=True)

        gen_loss = get_gen_loss(fake_value)
        disc_loss = get_disc_loss(real_value, fake_value)

    gen_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradient, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))

    return gen_loss, disc_loss

for epoch in range(EPOCHS):
    start_time = time.time()

    for batch, label in train_ds:
        gen_loss, disc_loss = train_step(batch, label)

    if (epoch+1) % 20 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
        display.clear_output(wait=True)
        generate_and_save_images(generator, label[0], epoch+1)

    print(f'epoch = {epoch+1} / time = {time.time()-start_time} / gen_loss = {gen_loss} / disc_loss = {disc_loss}')
