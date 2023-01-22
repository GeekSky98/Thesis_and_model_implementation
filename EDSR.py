import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Add, Conv2D, Input, Lambda, Rescaling
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

channel = 3
batch_size = 32
epoch = 100
AUTOTUNE = tf.data.AUTOTUNE

div2k_data = tfds.image.Div2k(config="bicubic_x4")
div2k_data.download_and_prepare()

train = div2k_data.as_dataset(split="train", as_supervised=True)
train_cache = train.cache()

val = div2k_data.as_dataset(split="validation", as_supervised=True)
val_cache = val.cache()

def flip_left_right(lowres_img, highres_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(
        rn < 0.5,
        lambda: (lowres_img, highres_img),
        lambda: (
            tf.image.flip_left_right(lowres_img),
            tf.image.flip_left_right(highres_img),
        ),
    )


def random_rotate(lowres_img, highres_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lowres_img, rn), tf.image.rot90(highres_img, rn)


def random_crop(lowres_img, highres_img, hr_crop_size=96, scale=4):
    lowres_crop_size = hr_crop_size // scale
    lowres_img_shape = tf.shape(lowres_img)[:2]

    lowres_width = tf.random.uniform(
        shape=(), maxval=lowres_img_shape[1] - lowres_crop_size + 1, dtype=tf.int32
    )
    lowres_height = tf.random.uniform(
        shape=(), maxval=lowres_img_shape[0] - lowres_crop_size + 1, dtype=tf.int32
    )

    highres_width = lowres_width * scale
    highres_height = lowres_height * scale

    lowres_img_cropped = lowres_img[
        lowres_height : lowres_height + lowres_crop_size,
        lowres_width : lowres_width + lowres_crop_size,
    ]
    highres_img_cropped = highres_img[
        highres_height : highres_height + hr_crop_size,
        highres_width : highres_width + hr_crop_size,
    ]

    return lowres_img_cropped, highres_img_cropped

def dataset_object(dataset_cache, training=True):

    ds = dataset_cache
    ds = ds.map(
        lambda lowres, highres: random_crop(lowres, highres, scale=4),
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
        ds = ds.map(flip_left_right, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)

    if training:
        ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = dataset_object(train_cache, training=True)
val_ds = dataset_object(val_cache, training=False)

class EDSR_model_builder(Model):
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        x = tf.cast(tf.expand_dims(data, axis=0), tf.float32)
        pred = self(x, training=False)
        pred = tf.clip_by_value(pred, 0, 255)
        pred = tf.round(pred)
        pred = tf.cast(tf.squeeze(pred, axis=0), tf.uint8)

        return pred

def res_block(Input_layer, filter=64):
    res_x = Conv2D(filters=filter, kernel_size=3, activation='relu', padding='same')(Input_layer)
    res_x = Conv2D(filters=filter, kernel_size=3, padding='same')(res_x)
    res_x = Lambda(function=lambda x: x * 0.1)(res_x)
    result = Add()([Input_layer, res_x])

    return result

def upsampling(Input_layer, filter=64, scale=2, **sky):
    up_x = Conv2D(filter * (scale ** 2), kernel_size=3, padding='same', **sky)(Input_layer)
    up_x = tf.nn.depth_to_space(up_x, block_size=scale)
    up_x = Conv2D(filter * (scale ** 2), kernel_size=3, padding='same', **sky)(up_x)
    up_x = tf.nn.depth_to_space(up_x, block_size=scale)

    return up_x

def build_edsr(filter=64, num_res=16):
    input = Input((None, None, channel))
    x = Rescaling(scale=1.0/255.0)(input)
    x = res_x = Conv2D(filter, kernel_size=3, padding='same')(x)

    for _ in range(num_res):
        res_x = res_block(res_x)

    res_x = Conv2D(filter, kernel_size=3, padding='same')(res_x)
    x = Add()([x, res_x])

    up_x = upsampling(x)
    output = Conv2D(channel, kernel_size=3, padding='same')(up_x)
    output = Rescaling(255)(output)

    model = EDSR_model_builder(input, output)

    return model

model = build_edsr()

oprimizer = keras.optimizers.Adam(
    learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[5000], values=[1e-4, 5e-5])
)

def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=255)[0]

def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=255)[0]

model.compile(
    loss = 'mae',
    optimizer=oprimizer,
    metrics=[PSNR, SSIM]
)


filename = 'checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(epoch, batch_size)
checkpoint = ModelCheckpoint(filename, monitor='PSNR', verbose=1, save_best_only=True, mode='auto')
earlystopping = EarlyStopping(monitor='PSNR', patience=20)


model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch,
    steps_per_epoch=300,
    callbacks=[checkpoint,earlystopping],
    verbose=1
)

def result(num):
    for lowres, highres in val.take(num):
        lowres = tf.image.random_crop(lowres, (150, 150, 3))
        pred = model.predict_step(lowres)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1), plt.imshow(lowres), plt.title("Low")
    plt.subplot(1, 3, 2), plt.imshow(pred), plt.title("EDSR")
    plt.subplot(1, 3, 3), plt.imshow(highres), plt.title("HR")
    plt.show()
    print(lowres.shape, pred.shape)

result(4)

model.save_weights('./model/EDSR_weight')

model2 = build_edsr()
model2.load_weights('./model/EDSR_weight')


import numpy as np
from PIL import Image
sample = Image.open('./data/sample.jpg')
sample = np.array(sample)
sample = tf.image.random_crop(sample, (150,150,3))
pred = model.predict_step(sample)

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1), plt.imshow(sample), plt.title("Low")
plt.subplot(1, 2, 2), plt.imshow(pred), plt.title("EDSR")
plt.show()