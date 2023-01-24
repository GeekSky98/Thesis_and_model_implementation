from keras.models import Sequential
from keras.layers import Conv2D
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.callbacks import EarlyStopping, ModelCheckpoint

batch_size = 32
epoch = 50
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

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=9, padding='same', activation='relu', input_shape=(None, None, 3)))
model.add(Conv2D(filters=32, kernel_size=1, padding='same', activation='relu'))
model.add(Conv2D(filters=3, kernel_size=5, padding='same', activation='sigmoid'))

model.summary()

def PSNR(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=255)[0]

model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['PSNR']
)

filename = 'checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(epoch, batch_size)
checkpoint = ModelCheckpoint(filename, monitor='PSNR', verbose=1, save_best_only=True, mode='auto')
earlystopping = EarlyStopping(monitor='PSNR', patience=20)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch,
    steps_per_epoch=300,
    verbose=1,
    callbacks=[checkpoint, earlystopping]
)