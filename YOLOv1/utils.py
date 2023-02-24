import tensorflow as tf


def iou(yolo_pred_boxes, ground_truth_boxes):
  boxes1 = yolo_pred_boxes
  boxes2 = ground_truth_boxes

  boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                     boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
  boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
  boxes2 = tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                     boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])
  boxes2 = tf.cast(boxes2, tf.float32)

  lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
  rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

  intersection = rd - lu

  inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

  mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)

  inter_square = mask * inter_square

  square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
  square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

  return inter_square / (square1 + square2 - inter_square + 1e-6)