import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from skimage import color
import numpy as np
from keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, Dropout
from keras.layers import BatchNormalization, Activation, MaxPool2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import gdown, zipfile, os, math

# Setting value
img_size = 150
n_train = 6500
n_val = 7106 - 6500
LR = 0.0001
n_batch = 32
n_epoch = 50

# Download data
url = "https://drive.google.com/uc?id=1-K9o_YAGJbQeyFPR9CXhRLcgyex6bVjy"

gdown.download(url, 'landscape_data.zip', quiet = False)

with zipfile.ZipFile('landscape_data.zip', 'r') as z_fp:
  z_fp.extractall('./data/landscape_data')

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, "data")
last_data_dir = os.path.join(data_dir, "landscape_data")

image_files = [file_name for file_name in os.listdir(last_data_dir) if os.path.splitext(file_name)[-1] == '.jpg']
len(image_files)

for path in image_files:
  temp = np.array(Image.open(os.path.join(last_data_dir, path))).shape
  if temp != (img_size, img_size, 3):
    os.remove(os.path.join(last_data_dir, path))
  else:
    continue

image_files = [file_name for file_name in os.listdir(last_data_dir) if os.path.splitext(file_name)[-1] == '.jpg']
len(image_files)

data = np.array([np.array(Image.open(os.path.join(last_data_dir,image)))for image in image_files])
print(f"data shape : {data.shape}")

data = data / data.max()

data_gray = np.expand_dims(np.array(color.rgb2gray(data)), -1)

train_x, val_x, train_y, val_y = train_test_split(data_gray, data, test_size = 0.2, shuffle = True, random_state = 777)

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

train_ds = Dataloader(train_x, train_y, n_batch, shuffle = True)
validation_ds = Dataloader(val_x, val_y, n_batch)

def conv2d_block(x, channel):
  x = Conv2D(channel, 3, padding="same")(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  x = Conv2D(channel, 3, padding="same")(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  return x

def unet():
  inputs = Input((150, 150, 1))

  con_1 = conv2d_block(inputs, 32)
  pool_1 = MaxPool2D((2, 2))(con_1)
  pool_1 = Dropout(0.1)(pool_1)

  con_2 = conv2d_block(pool_1, 64)
  pool_2 = MaxPool2D((2, 2))(con_2)
  pool_2 = Dropout(0.1)(pool_2)

  con_3 = conv2d_block(pool_2, 128)
  pool_3 = MaxPool2D((2, 2))(con_3)
  pool_3 = Dropout(0.1)(pool_3)

  con_4 = conv2d_block(pool_3, 256)
  pool_4 = MaxPool2D((2, 2))(con_4)
  pool_4 = Dropout(0.1)(pool_4)

  con_5 = conv2d_block(pool_4, 512)

  unite_1 = Conv2DTranspose(256, 2, 2, output_padding=(0, 0))(con_5)
  unite_1 = concatenate([unite_1, con_4])
  unite_1 = Dropout(0.1)(unite_1)
  unite_1_con = conv2d_block(unite_1, 256)

  unite_2 = Conv2DTranspose(128, 2, 2, output_padding=(1, 1))(unite_1_con)
  unite_2 = concatenate([unite_2, con_3])
  unite_2 = Dropout(0.1)(unite_2)
  unite_2_con = conv2d_block(unite_2, 128)

  unite_3 = Conv2DTranspose(64, 2, 2, output_padding=(1, 1))(unite_2_con)
  unite_3 = concatenate([unite_3, con_2])
  unite_3 = Dropout(0.1)(unite_3)
  unite_3_con = conv2d_block(unite_3, 64)

  unite_4 = Conv2DTranspose(32, 2, 2, output_padding=(0, 0))(unite_3_con)
  unite_4 = concatenate([unite_4, con_1])
  unite_4 = Dropout(0.1)(unite_4)
  unite_4_con = conv2d_block(unite_4, 32)

  outputs = Conv2D(3, 1, activation="sigmoid")(unite_4_con)

  model = Model(inputs, outputs)
  return model

model_unet_graytocolor = unet()

class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, max_lr, warmup_steps, decay_steps):
    super(CustomSchedule, self).__init__()
    self.max_lr = max_lr
    self.warmup_steps = warmup_steps
    self.decay_steps = decay_steps

  def __call__(self, step):
    lr = tf.cond(step < self.warmup_steps,
                 lambda: self.max_lr / self.warmup_steps * step,
                 lambda: 0.5 * (1 + tf.math.cos(math.pi * (step - self.warmup_steps) / self.decay_steps)) * self.max_lr)
    return lr

steps_per_epoch = n_train // n_batch
lr_schedule = CustomSchedule(LR, 3 * steps_per_epoch, n_epoch * steps_per_epoch)

model_unet_graytocolor.compile(
  loss = "mae",
  optimizer = keras.optimizers.Adam(lr_schedule),
  metrics = "accuracy"
)

model_unet_graytocolor.fit(
  train_ds,
  validation_data = validation_ds,
  epochs = n_epoch,
  verbose = 1
)

model_unet_graytocolor.save("./model/model_unet_graytocolor.h5", include_optimizer = False)

model_unet_graytocolor.evaluate(validation_ds, verbose = 1)


def contrast_image(index):
  res = model_unet_graytocolor.predict(np.expand_dims(train_x[index], 0)).reshape(img_size, img_size, 3)
  plt.imshow(np.concatenate([res, train_y[index]], axis=1).reshape(img_size, -1, 3))
  plt.colorbar()
  plt.title("result / original image")
  plt.show()

model_unet_graytocolor = tf.keras.models.load_model("./model/model_unet_graytocolor.h5")

model_unet_graytocolor.compile(
  loss = "mae",
  optimizer = keras.optimizers.Adam(lr_schedule),
  metrics = "accuracy"
)

def contrast(index):

  plt.subplot(1, 3, 1)
  plt.imshow(train_x[index], cmap='gray')
  plt.title("Gray Scale Image")
  plt.subplot(1, 3, 2)
  plt.imshow(model_unet_graytocolor.predict(np.expand_dims(train_x[index], 0)).reshape(150, 150, 3))
  plt.title("Gray To Color")
  plt.subplot(1, 3, 3)
  plt.imshow(train_y[index])
  plt.title("Color Image")

  plt.tight_layout()
  plt.show()


contrast(113)