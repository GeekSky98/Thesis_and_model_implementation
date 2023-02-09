import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import IPython.display as display
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
import time, os

hr_img_size = 100
lr_img_size = 25
batch_size = 16
epochs = 100
lr = 1e-4
AUTOTUNE = tf.data.AUTOTUNE


cur_dir = os.getcwd()

cgan_path = os.path.join(cur_dir, 'cgan')
checkpoint_dir = os.path.join(cgan_path, 'checkpoint')
output_dir = os.path.join(cgan_path, 'output')

train, val = tfds.load("div2k/bicubic_x4", split = ['train', 'validation'], as_supervised=True)

'''
def preprocessing(low_image, high_image):
    high_scaling = tf.cast(high_image / 255, tf.float32)
    high_crop = tf.image.random_crop(high_scaling, size=(hr_img_size, hr_img_size, 3))
    low_crop = tf.image.resize(high_crop, (lr_img_size, lr_img_size), method="bicubic")

    return low_crop, high_crop

train_ds = train.map(preprocessing).shuffle(buffer_size=10).repeat().batch(batch_size).prefetch(AUTOTUNE)
val_ds = val.map(preprocessing).batch(1).prefetch(AUTOTUNE)'''


def preprocessing2(low, high):
    # high_img = tf.cast(tf.image.random_crop(high, size=(hr_img_size, hr_img_size, 3)), tf.float32)
    high_img = tf.cast(tf.image.resize(high, [hr_img_size,hr_img_size]), tf.float32)
    low_img = tf.image.resize(high_img, [lr_img_size, lr_img_size])
    high_img, low_img = (high_img - 127.5) / 127.5, (low_img - 127.5) / 127.5
    return low_img, high_img

train_ds = train.map(preprocessing2).shuffle(buffer_size=len(train)).batch(batch_size).prefetch(AUTOTUNE)
val_ds = val.map(preprocessing2).batch(1).prefetch(AUTOTUNE)

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

    outputs = Conv2D(filters=3, kernel_size=9, strides=1, padding='same', activation='tanh')(x)

    return Model(inputs, outputs)

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

    return Model(inputs, outputs)

vgg_transfer = vgg.VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
vgg_transfer.summary()

vgg_model = Model(vgg_transfer.input, vgg_transfer.get_layer('block5_conv4').output)

generator = build_generator()
discriminator = build_discriminator()
mse = MeanSquaredError()
binaryCE = BinaryCrossentropy()

def get_generator_loss(fake_output):
    return binaryCE(tf.ones_like(fake_output), fake_output)

def get_discriminator_loss(real_output, fake_output):
    return binaryCE(tf.ones_like(real_output), real_output) + \
           binaryCE(tf.zeros_like(fake_output), fake_output)

def get_content_loss_vgg(hr_real, hr_fake):
    hr_real, hr_fake = vgg.preprocess_input(hr_real), vgg.preprocess_input(hr_fake)

    hr_real_feature_map = vgg_model(hr_real) / 12.75
    hr_fake_feature_map = vgg_model(hr_fake) / 12.75

    return mse(hr_real_feature_map, hr_fake_feature_map)

generator_optimizer = optimizers.Adam(learning_rate=lr)
discriminator_optimizer = optimizers.Adam(learning_rate=lr)


checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


def to_range(images, min_value=0, max_value=255):
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(np.uint8)

def generate_and_save_images(model, low, high, epoch):

    pred = model(low, training=False)
    # image = np.reshape(image, [200, 200, 3])
    image_list = [low, pred, high]
    image_name = ["Low image", "SRGAN result", "High image"]
    for i, image in enumerate(image_list):
        image = tf.squeeze(image, 0)
        image = np.array(image)
        image = to_range(image)

        plt.subplot(1, 3, i+1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(image_name[i])
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, 'image_at_epoch_{:05d}.png'.format(epoch)))


def train_step(low_img, high_real):
    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
        fake = generator(low_img, training=True)

        real_out = discriminator(high_real, training=True)
        fake_out = discriminator(fake, training=True)

        perceptual_loss = get_content_loss_vgg(high_real, fake) + 1e-3 * get_generator_loss(fake_out)
        discriminator_loss = get_discriminator_loss(real_out, fake_out)

    generator_gradient = generator_tape.gradient(perceptual_loss, generator.trainable_variables)
    discriminator_gradient = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))

    return perceptual_loss, discriminator_loss

test_iter = iter(val_ds)

for epoch in range(epochs):
    start_time = time.time()

    for low_img, high_img in train_ds:
        g_loss, d_loss = train_step(low_img, high_img)

    if (epoch+1) % 3 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
        display.clear_output(wait=True)
        low, high = next(test_iter)
        generate_and_save_images(generator, low, high, epoch+1)

    print(f'epoch = {epoch + 1} / time = {time.time() - start_time} / gen_loss = {g_loss} / disc_loss = {d_loss}')


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
    lr_image = resize(hr_image, (img_height//4, img_width//4, 3))
    srgan_image = result(lr_image)
    bicubic_image = resize(lr_image, (img_height, img_width, 3))

    plt.imshow(np.concatenate([bicubic_image, srgan_image, hr_image], axis=1))
    plt.show()

from PIL import Image
import cv2

temp = Image.open('./data/sample.jpg')

contrast(temp)

#generator.evaluate(val_ds)

temp = np.array(temp)
temp = resize(temp, (25, 25, 3))

temp = tf.cast(tf.expand_dims(temp, 0), tf.float32)

res = generator.predict(temp)
res2 = tf.clip_by_value(res, 0, 255)
res2 = tf.round(res2)
res2 = tf.cast(res2, tf.uint8)
plt.imshow(tf.squeeze(res, 0))
plt.show()

img = cv2.imread('./data/sample.jpg',1)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_resize = cv2.resize(img,dsize=(0,0),fx=0.25,fy=0.25,interpolation=cv2.INTER_CUBIC)
bicubic_hr = cv2.resize(img_resize,dsize=(0,0),fx=4,fy=4,interpolation=cv2.INTER_CUBIC)

def apply_srgan(image):
    image = tf.cast(image[np.newaxis, ...], tf.float32)
    sr = generator.predict(image)
    sr = tf.clip_by_value(sr, 0, 255)
    sr = tf.round(sr)
    sr = tf.cast(sr, tf.uint8)
    return np.array(sr)[0]

plt.imshow(apply_srgan(img_resize))
plt.show()

a = [1, 2, 4]
for i, num in enumerate(a):
    print(num)


ran = tf.random.normal(shape=(10,10,3))
ran2 = ran
ran3 = ran

a_list = [ran, ran2, ran3]

for i, image in enumerate(a_list):
    #image = tf.squeeze(image, 0)
    #image = to_range(image)

    plt.subplot(1, 3, i + 1)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

ran