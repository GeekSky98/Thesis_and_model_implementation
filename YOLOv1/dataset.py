import tensorflow as tf
import numpy as np


def bounds_per_dimension(ndarray):
  return map(lambda x: range(x.min(), x.max() + 1), np.where(ndarray != 0))


def zero_trim_ndarray(ndarray):
  return ndarray[np.ix_(*bounds_per_dimension(ndarray))]


def process_each_ground_truth(original_image,
                              bbox,
                              class_labels,
                              input_width,
                              input_height
                              ):
  image = original_image.numpy()
  image = zero_trim_ndarray(image)

  original_h = image.shape[0]
  original_w = image.shape[1]

  width_rate = input_width * 1.0 / original_w
  height_rate = input_height * 1.0 / original_h

  image = tf.image.resize(image, [input_height, input_width])

  object_num = np.count_nonzero(bbox, axis=0)[0]
  labels = [[0, 0, 0, 0, 0]] * object_num
  for i in range(object_num):
    xmin = bbox[i][1] * original_w
    ymin = bbox[i][0] * original_h
    xmax = bbox[i][3] * original_w
    ymax = bbox[i][2] * original_h

    class_num = class_labels[i]

    xcenter = (xmin + xmax) * 1.0 / 2 * width_rate
    ycenter = (ymin + ymax) * 1.0 / 2 * height_rate

    box_w = (xmax - xmin) * width_rate
    box_h = (ymax - ymin) * height_rate

    labels[i] = [xcenter, ycenter, box_w, box_h, class_num]

  return [image.numpy(), labels, object_num]