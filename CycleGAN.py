import functools
import tqdm
from experiment_utils import *
import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, BatchNormalization, Activation, add, Input, Conv2DTranspose, ZeroPadding2D, LeakyReLU
from tensorflow_addons.layers import InstanceNormalization
from keras.models import Model
from PIL import Image
import os, random
from keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras import optimizers
import IPython.display as display
import imageio
from glob import glob
import IPython
# %load_ext tensorboard

input_shape = (128, 128, 3)
hidden_layers = 3
n_batch = 1
LR = 2e-4
BETA_1 = 0.5
epoch_decay = 100
epochs = 125
cycle_loss_weight = 10.0
identity_loss_weight = 0.0
gradient_penalty_weight = 10.0
gradient_penalty_mode = 'none'
autotune = tf.data.AUTOTUNE

cur_path = os.getcwd()
data_path = os.path.join(cur_path, 'data\\gan\\vangogh2photo\\vangogh2photo')

train_a_path = os.path.join(data_path, 'trainA')
train_b_path = os.path.join(data_path, 'trainB')
test_a_path = os.path.join(data_path, 'testA')
test_b_path = os.path.join(data_path, 'testB')

cyclegan_path = os.path.join(cur_path, 'cyclegan')
checkpoint_dir = os.path.join(cyclegan_path, 'checkpoint')
output_dir = os.path.join(cyclegan_path, 'output')
sample_dir = os.path.join(output_dir, 'sample_training')
summary_train_dir = os.path.join(output_dir, 'summaries', 'train')
train_summary_writer = tf.summary.create_file_writer(summary_train_dir)

def get_image(path):
    dir = [file_name for file_name in os.listdir(path) if os.path.splitext(file_name)[-1] == '.jpg']
    image = np.array([np.array(Image.open(os.path.join(path, img))) for img in dir])
    print(f'image length : {len(image)}')
    return image

train_a = get_image(train_a_path)
train_b = get_image(train_b_path)
test_a = get_image(test_a_path)
test_b = get_image(test_b_path)

print(train_a.shape, train_b.shape, test_a.shape, test_b.shape)
print(train_a.max(), train_b.max(), test_a.max(), test_b.max())
print(train_a.min(), train_b.min(), test_a.min(), test_b.min())

ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
seed = random.seed(7777777)

def preprocessing(data):
    image = tf.cast(data, tf.uint8)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, [128, 128])
    image = (image-127.5) / 127.5
    return image

def custom_dataloader(data, mode):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.map(preprocessing, num_parallel_calls=autotune)
    if mode == 'test':
        dataset = dataset.repeat(n_batch * epochs)
    dataset = dataset.batch(n_batch)
    dataset = dataset.prefetch(autotune)

    return dataset

train_a = custom_dataloader(train_a, mode='train')
train_b = custom_dataloader(train_b, mode='train')
test_a = custom_dataloader(test_a, mode='test')
test_b = custom_dataloader(test_b, mode='test')

train_ds = tf.data.Dataset.zip((train_a, train_b))
test_ds = tf.data.Dataset.zip((test_a, test_b))
len_dataset = max(len(train_a), len(train_b)) // n_batch

def residual_block(input_layer):

    x = input_layer
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(axis=3, momentum=0.9, epsilon=1e-5)(x)

    return add([input_layer, x])

def InstanceNorm(input_layer, filter, kernel_size, stride):

    x = Conv2D(filters=filter, kernel_size=kernel_size, strides=stride, padding='same')(input_layer)
    x = InstanceNormalization(axis=1)(x)
    x = Activation('relu')(x)

    return x

def Transpose(input_layer, filter, kernel_size, stride, use_bias=False):

    x = Conv2DTranspose(filters=filter, kernel_size=kernel_size, strides=stride, padding='same',
                        use_bias=use_bias)(input_layer)
    x = InstanceNormalization(axis=1)(x)
    x = Activation('relu')(x)

    return x

def generator():

    input = Input(input_shape)

    x = InstanceNorm(input, 32, 7, 1)
    x = InstanceNorm(x, 64, 3, 2)
    x = InstanceNorm(x, 128, 3, 2)

    for _ in range(6):
        x = residual_block(x)

    x = Transpose(x, 64, 3, 2)
    x = Transpose(x, 32, 3, 2)

    x = Conv2D(filters=3, kernel_size=7, strides=1, padding='same')(x)
    output = Activation('tanh')(x)

    return Model(input, output)

def discriminator():

    input = Input(input_shape)

    x = ZeroPadding2D(padding=(1, 1))(input)
    x = Conv2D(filters=64, kernel_size=4, strides=2, padding='valid')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    for i in range(1, hidden_layers + 1):
        x = Conv2D(filters=2 ** i * 64, kernel_size=4, strides=2, padding='valid')(x)
        x = InstanceNormalization(axis=1)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = ZeroPadding2D(padding=(1, 1))(x)

    output = Conv2D(filters=1, kernel_size=4, strides=1, activation='sigmoid')(x)

    return Model(input, output)

Gen_a_to_b = generator()
Gen_b_to_a = generator()

Disc_a = discriminator()
Disc_b = discriminator()

mse = MeanSquaredError()

def get_gen_loss(fake):
    f_loss = mse(tf.ones_like(fake), fake)
    return f_loss

def get_disc_loss(real, fake):
    r_loss = mse(tf.ones_like(real), real)
    f_loss = mse(tf.zeros_like(fake), fake)
    return r_loss, f_loss

cycle_loss = MeanAbsoluteError()
identity_loss = MeanAbsoluteError()


def gradient_penalty(tool, real, fake):
    def interpolate(a, b):
        shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
        alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.shape)
        return inter

    x = interpolate(real, fake)
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = tool(x)
    gradient = tape.gradient(pred, x)
    norm = tf.norm(tf.reshape(gradient, [tf.shape(gradient)[0], -1]), axis=1)
    output = tf.reduce_mean((norm - 1.) ** 2)

    return output

gen_lr_scheduler = LinearDecay(LR, epochs * len_dataset, epoch_decay * len_dataset)
disc_lr_scheduler = LinearDecay(LR, epochs * len_dataset, epoch_decay * len_dataset)

gen_optimizer = optimizers.Adam(learning_rate=gen_lr_scheduler, beta_1=BETA_1)
disc_optimizer = optimizers.Adam(learning_rate=disc_lr_scheduler, beta_1=BETA_1)

def train_gen(A, B):
    with tf.GradientTape() as tape:
        A2B = Gen_a_to_b(A, training=True)
        B2A = Gen_b_to_a(B, training=True)
        A2B2A = Gen_b_to_a(A2B, training=True)
        B2A2B = Gen_a_to_b(B2A, training=True)
        A2A = Gen_b_to_a(A, training=True)
        B2B = Gen_a_to_b(B, training=True)

        disc_A2B_value = Disc_b(A2B, training=True)
        disc_B2A_value = Disc_a(B2A, training=True)

        gen_loss_A2B = get_gen_loss(disc_A2B_value)
        gen_loss_B2A = get_gen_loss(disc_B2A_value)

        cycle_loss_A2B2A = cycle_loss(A, A2B2A)
        cycle_loss_B2A2B = cycle_loss(B, B2A2B)

        identity_loss_A2A = identity_loss(A, A2A)
        identity_loss_B2B = identity_loss(B, B2B)

        generator_loss = (gen_loss_A2B+gen_loss_B2A) + (cycle_loss_A2B2A+cycle_loss_B2A2B) * cycle_loss_weight + \
                   (identity_loss_A2A+identity_loss_B2B)*identity_loss_weight

    generator_gradient = tape.gradient(generator_loss, Gen_a_to_b.trainable_variables + Gen_b_to_a.trainable_variables)
    gen_optimizer.apply_gradients(zip(generator_gradient, Gen_a_to_b.trainable_variables +
                                      Gen_b_to_a.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': gen_loss_A2B,
                      'B2A_g_loss': gen_loss_B2A,
                      'A2B2A_cycle_loss': cycle_loss_A2B2A,
                      'B2A2B_cycle_loss': cycle_loss_B2A2B,
                      'A2A_id_loss': identity_loss_A2A,
                      'B2B_id_loss': identity_loss_B2B}

def train_disc(A, B, A2B, B2A):
    with tf.GradientTape() as tape:
        A_value = Disc_a(A, training=True)
        B2A_value = Disc_a(B2A, training=True)
        B_value = Disc_b(B, training=True)
        A2B_value = Disc_b(A2B, training=True)

        disc_loss_A, disc_loss_B2A = get_disc_loss(A_value, B2A_value)
        disc_loss_B, disc_loss_A2B = get_disc_loss(B_value, A2B_value)

        discriminator_A_GP = gradient_penalty(functools.partial(Disc_a, training=True), A, B2A)
        discriminator_B_GP = gradient_penalty(functools.partial(Disc_b, training=True), B, A2B)

        discriminator_loss = (disc_loss_A+disc_loss_B2A) + (disc_loss_B+disc_loss_A2B) + \
                             (discriminator_A_GP+discriminator_B_GP)*gradient_penalty_weight

        discriminator_gradient = tape.gradient(discriminator_loss, Disc_a.trainable_variables+
                                               Disc_b.trainable_variables)
        disc_optimizer.apply_gradients(zip(discriminator_gradient, Disc_a.trainable_variables+
                                           Disc_b.trainable_variables))

        return {'A_d_loss': disc_loss_A + disc_loss_B2A,
            'B_d_loss': disc_loss_B + disc_loss_A2B,
            'D_A_gp': discriminator_A_GP,
            'D_B_gp': discriminator_B_GP}

A2B_pool = ItemPool(5)
B2A_pool = ItemPool(5)

def train_step(A, B):
    A2B, B2A, gen_loss_total = train_gen(A, B)

    A2B = A2B_pool(A2B)
    B2A = B2A_pool(B2A)

    disc_loss_total = train_disc(A, B, A2B, B2A)

    return gen_loss_total, disc_loss_total

def sample_image(A, B):
    A2B = Gen_a_to_b(A, training=False)
    B2A = Gen_b_to_a(B, training=False)
    A2B2A = Gen_b_to_a(A2B, training=False)
    B2A2B = Gen_a_to_b(B2A, training=False)
    return A2B, B2A, A2B2A, B2A2B

test_iter = iter(test_ds)

checkpoint = Checkpoint(dict(G_A2B=Gen_a_to_b,
                                G_B2A=Gen_b_to_a,
                                D_A=Disc_a,
                                D_B=Disc_b,
                                G_optimizer=gen_optimizer,
                                D_optimizer=disc_optimizer,
                                ep_cnt=ep_cnt),
                           checkpoint_dir,
                           max_to_keep=5)

with train_summary_writer.as_default():
    for ep in tqdm.trange(epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        ep_cnt.assign_add(1)

        for A, B in tqdm.tqdm(train_ds, desc='Inner Epoch Loop', total=len_dataset):
            G_loss_dict, D_loss_dict = train_step(A, B)

            summary(G_loss_dict, step=gen_optimizer.iterations, name='generator_losses')
            summary(D_loss_dict, step=gen_optimizer.iterations, name='discriminator_losses')
            summary({'learning rate': gen_lr_scheduler.current_learning_rate}, step=gen_optimizer.iterations,
                    name='learning rate')

            if gen_optimizer.iterations.numpy() % 100 == 0:
                A, B = next(test_iter)
                A2B, B2A, A2B2A, B2A2B = sample_image(A, B)
                img = immerge(np.concatenate([A, A2B, A2B2A, B, B2A, B2A2B], axis=0), n_rows=2)
                imwrite(img, os.path.join(sample_dir, 'iter-%09d.jpg' % gen_optimizer.iterations.numpy()))

        checkpoint.save(ep)