import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage.transform import resize
import numpy as np
import os, math
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence

batch_size = 32
epoch = 50
AUTOTUNE = tf.data.AUTOTUNE

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, "data")
last_data_dir = os.path.join(data_dir, "landscape_data")

image_files = [file_name for file_name in os.listdir(last_data_dir) if os.path.splitext(file_name)[-1] == '.jpg']
len(image_files)

data = np.array([np.array(Image.open(os.path.join(last_data_dir,image)))for image in image_files])
print(f"data shape : {data.shape}")

data = data / data.max()

data_small = np.array([resize(img, (50, 50, 3)) for img in data])

data_small_bicu = np.array([resize(img, (150, 150, 3)) for img in data_small])

plt.imshow(np.concatenate([data[0], data_small_bicu[0]]))
plt.show()

train_x, val_x, train_y, val_y = train_test_split(data_small_bicu, data, test_size = 0.2, shuffle = True,
                                                  random_state = 777)

class Dataloader(Sequence):

  def __init__(self, data_x, data_y, batch_size, shuffle=False):
    self.x, self.y = data_x, data_y
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.on_epoch_end()

  def __len__(self):
    return math.ceil(len(self.x) / self.batch_size)

  def __getitem__(self, idx):
    indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

    batch_x = [self.x[i] for i in indices]
    batch_y = [self.y[i] for i in indices]

    return np.array(batch_x), np.array(batch_y)

  def on_epoch_end(self):
    self.indices = np.arange(len(self.x))
    if self.shuffle == True:
      np.random.shuffle(self.indices)

train_ds = Dataloader(train_x, train_y, batch_size, shuffle = True)
validation_ds = Dataloader(val_x, val_y, batch_size)

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
    metrics=[PSNR]
)

filename = 'checkpoint2-epoch-{}-batch-{}-trial-001.h5'.format(epoch, batch_size)
checkpoint = ModelCheckpoint(filename, monitor='PSNR', verbose=1, save_best_only=True, mode='auto')
earlystopping = EarlyStopping(monitor='PSNR', patience=20)

model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epoch,
    verbose=1,
    callbacks=[checkpoint, earlystopping]
)


def result(num):
    low = val_x[num]
    high = val_y[num]
    pred = model.predict(tf.expand_dims(low, 0))

    plt.imshow(np.concatenate([low, tf.squeeze(pred, 0), high], axis=-2))
    plt.show()

result(20)

model.save('./model/SRCNN.h5')