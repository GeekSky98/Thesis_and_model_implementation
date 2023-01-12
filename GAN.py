import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape, LeakyReLU
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam

img_size = 28
channels = 1
img_shape = (img_size, img_size, channels)

z = 100

# Create generator
def build_generator(img_shape, z):
    model = Sequential()

    model.add(Dense(128, input_dim = z))
    model.add(LeakyReLU(alpha = 0.01))
    model.add(Dense(28 * 28 * 1, activation = 'tanh'))
    model.add(Reshape(img_shape))

    return model

# Create discriminator
def build_discriminator(img_shape):
    model = Sequential()

    model.add(Flatten(input_shape = img_shape))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha = 0.01))
    model.add(Dense(1, activation = 'sigmoid'))

    return model

# combine G & D
def build_gan(genrator, discriminator):
    model = Sequential()

    model.add(genrator)
    model.add(discriminator)

    return model

discriminator = build_discriminator(img_shape)
discriminator.compile(
    loss = 'binary_crossentropy',
    optimizer = Adam(),
    metrics = ["accuracy"]
)

generator = build_generator(img_shape, z)

discriminator.trainable = False

gan = build_gan(generator, discriminator)
gan.compile(
    loss = "binary_crossentropy",
    optimizer = Adam()
)

losses = []
accuracies = []
iteration_checkpoint = []

(train_x, _), (_,_) = mnist.load_data()
train_x = train_x / 127.5 - 1.0
train_x = np.expand_dims(train_x, -1)

def train(iterations, batch_size, sample_interval):
    real_image_label = np.ones((batch_size, 1))
    fake_image_label = np.zeros((batch_size, 1))

    for iteration in range(iterations):
        idx = np.random.randint(0, train_x.shape[0], batch_size)
        image = train_x[idx]

        z_image = np.random.normal(0, 1, (batch_size, 100))
        gen_image = generator.predict(z_image)

        d_loss_real = discriminator.train_on_batch(image, real_image_label)
        d_loss_fake = discriminator.train_on_batch(gen_image, fake_image_label)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        g_loss = gan.train_on_batch(z_image, real_image_label)

        if(iteration + 1) % sample_interval == 0:
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoint.append(iteration + 1)

            print("%d [D손실 : %f, 정확도 : %.2f%%] [G 손실 : %f]" %
                  (iteration + 1, d_loss, 100.0 * accuracy, g_loss))

            sample_images(generator)

def sample_images(generator, image_grid_rows=4, image_grid_columns=4):

    z_random = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z))

    gen_imgs = generator.predict(z_random)

    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(image_grid_rows,
                            image_grid_columns,
                            figsize=(4, 4),
                            sharey=True,
                            sharex=True)

    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1

iterations = 20000
batch_size = 128
sample_interval = 1000

train(iterations, batch_size, sample_interval)

plt.imshow(generator.predict(np.random.normal(0, 1, (1, z)))[0], cmap="gray")
plt.show()