import sys
sys.path.append('C:/Users/whgks/PycharmProjects/model/YOLOv1')
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as display
import os
from loss import yolo_loss
from model import YOLOv1
from dataset import process_each_ground_truth
from utils import generate_color, yolo_format_to_bounding_box_dict, find_max_confidence_bounding_box, draw_bounding_box_and_label_info
import time, datetime
import cv2

validation_steps = 50
num_epochs = 135
init_learning_rate = 0.0001
lr_decay_rate = 0.5
lr_decay_steps = 2000
num_visualize_image = 8

batch_size = 128
test_batch_size = 1
input_width = 224
input_height = 224
cell_size = 7
num_classes = 1
boxes_per_cell = 2

coord_scale = 10
class_scale = 0.1
object_scale = 1
noobject_scale = 0.5

label_dict = {
  0:"person"
}
class_to_label_dict = {v: k for k, v in label_dict.items()}

color_list = generate_color(num_classes)

voc2007_test_split_data = tfds.load("voc/2007", split=tfds.Split.TEST, batch_size=1)
voc2012_train_split_data = tfds.load("voc/2012", split=tfds.Split.TRAIN, batch_size=1)
voc2012_validation_split_data = tfds.load("voc/2012", split=tfds.Split.VALIDATION, batch_size=1)
train_data = voc2007_test_split_data.concatenate(voc2012_train_split_data).concatenate(voc2012_validation_split_data)
n_data = round(len(train_data))

test_data = tfds.load("voc/2007", split=tfds.Split.TRAIN, batch_size=1)

def predicate(x, allowed_labels=tf.constant([14.0])):
  label = x['objects']['label']
  isallowed = tf.equal(allowed_labels, tf.cast(label, tf.float32))
  reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))

  return tf.greater(reduced, tf.constant(0.))

train_data = train_data.filter(predicate)
train_data = train_data.padded_batch(batch_size)

test_data = test_data.filter(predicate)
test_data = test_data.padded_batch(test_batch_size)

def reshape_yolo_preds(preds):
  return tf.reshape(preds, [tf.shape(preds)[0], cell_size, cell_size, num_classes + 5 * boxes_per_cell])


def calculate_loss(model, batch_image, batch_bbox, batch_labels):
  total_loss = 0.0
  coord_loss = 0.0
  object_loss = 0.0
  noobject_loss = 0.0
  class_loss = 0.0
  for batch_index in range(batch_image.shape[0]):
    image, labels, object_num = process_each_ground_truth(batch_image[batch_index], batch_bbox[batch_index], batch_labels[batch_index], input_width, input_height)
    image = tf.expand_dims(image, axis=0)

    predict = model(image)
    predict = reshape_yolo_preds(predict)

    for object_num_index in range(object_num):
      each_object_total_loss, each_object_coord_loss, each_object_object_loss, each_object_noobject_loss, each_object_class_loss = yolo_loss(predict[0],
                                   labels,
                                   object_num_index,
                                   num_classes,
                                   boxes_per_cell,
                                   cell_size,
                                   input_width,
                                   input_height,
                                   coord_scale,
                                   object_scale,
                                   noobject_scale,
                                   class_scale
                                   )

      total_loss = total_loss + each_object_total_loss
      coord_loss = coord_loss + each_object_coord_loss
      object_loss = object_loss + each_object_object_loss
      noobject_loss = noobject_loss + each_object_noobject_loss
      class_loss = class_loss + each_object_class_loss

  return total_loss, coord_loss, object_loss, noobject_loss, class_loss

def test_function(model):
    for data in test_data.take(1):
        image = tf.squeeze(data['image'], axis=1)
        bbox = tf.squeeze(data['objects']['bbox'], axis=1)
        labels = tf.squeeze(data['objects']['label'], axis=1)

        image, bbox, labels = process_each_ground_truth(image[0], bbox[0], labels[0], input_width, input_height)

        pred = reshape_yolo_preds(model(tf.expand_dims(image, 0)))
        pred_bbox = pred[0, :, :, num_classes+boxes_per_cell:]
        pred_bbox = tf.reshape(pred_bbox, [cell_size, cell_size, boxes_per_cell, 4])
        pred_confidence = pred[0, :, :, num_classes:num_classes+boxes_per_cell]
        pred_confidence = tf.reshape(pred_confidence, [cell_size, cell_size, boxes_per_cell, 1])

        pred_class = tf.argmax(pred[0, :, :, :num_classes], axis=2)

        bounding_box_list = []
        for i in range(cell_size):
            for j in range(cell_size):
                for k in range(boxes_per_cell):
                    pred_xcenter = pred_bbox[i][j][k][0]
                    pred_ycenter = pred_bbox[i][j][k][1]
                    pred_width = tf.minimum(input_width * 1.0, tf.maximum(0.0, pred_bbox[i][j][k][2]))
                    pred_height = tf.minimum(input_height * 1.0, tf.maximum(0.0, pred_bbox[i][j][k][3]))

                    pred_class_name = label_dict[pred_class[i][j].numpy()]
                    pred_confidence_value = pred_confidence[i][j][k].numpy()

                    bounding_box_list.append(
                        yolo_format_to_bounding_box_dict(pred_xcenter, pred_ycenter, pred_width, pred_height,
                                                         pred_class_name, pred_confidence_value)
                    )

        bounding_box = find_max_confidence_bounding_box(bounding_box_list)

        draw_bounding_box_and_label_info(
            image,
            bounding_box['xmin'],
            bounding_box['ymin'],
            bounding_box['xmax'],
            bounding_box['ymax'],
            bounding_box['class'],
            bounding_box['confidence'],
            color_list[class_to_label_dict[bounding_box['class']]]
        )

        plt.imshow(image.astype(np.uint8))
        plt.title(f'{epoch}epoch result')
        plt.show()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
  init_learning_rate,
  decay_steps=lr_decay_steps,
  decay_rate=lr_decay_rate,
  staircase=True)

optimizer = tf.optimizers.Adam(lr_schedule)

model_yolo = YOLOv1(input_height, input_width, cell_size, boxes_per_cell, num_classes)

checkpoint_prefix = os.path.join('./checkpoint', 'ckpt')
checkpoint = tf.train.Checkpoint(model = model_yolo)

def train_step(model, optimizer, batch_image, batch_bbox, batch_labels):
  with tf.GradientTape() as tape:
    total_loss, coord_loss, object_loss, noobject_loss, class_loss = calculate_loss(model, batch_image,
                                                                                    batch_bbox, batch_labels)

  gradients = tape.gradient(total_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return total_loss, coord_loss, object_loss, noobject_loss, class_loss

iteration_per_epoch = n_data/batch_size
for epoch in range(num_epochs):
    start_time = time.time()
    for iteration, features in enumerate(train_data):
        batch_image = features['image']
        batch_bbox = features['objects']['bbox']
        batch_labels = features['objects']['label']

        batch_image = tf.squeeze(batch_image, axis=1)
        batch_bbox = tf.squeeze(batch_bbox, axis=1)
        batch_labels = tf.squeeze(batch_labels, axis=1)

        total_loss, coord_loss, object_loss, noobject_loss, class_loss = train_step(model_yolo, optimizer, batch_image,
                                                                                    batch_bbox, batch_labels)

        if (iteration + 1) % 30 == 0:
            spent_time = str(datetime.timedelta(seconds=round(time.time() - start_time)))
            least_iteration = iteration_per_epoch-iteration
            print(f'iteration={iteration}, Time spent={spent_time}, least iteration={least_iteration}')

    if (epoch + 1) % 5 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
        display.clear_output(wait=True)
        test_function(model_yolo)

    TIME = time.time() - start_time
    EXPECT = str(datetime.timedelta(seconds=round((num_epochs - (epoch + 1)) * TIME)))
    print(f'epoch = {epoch + 1} / time = {TIME} / total_loss = {total_loss} / expect = {EXPECT}')
    print(f'coord_loss={coord_loss}, object_loss={object_loss}, noobject_loss={noobject_loss}, class_loss={class_loss}')


model_yolo.save_weights('./YOLOv1/saved_model/yolo2.h5')

test_function(model_yolo)

model_yolo.load_weights("./saved_model/yolo.h5")