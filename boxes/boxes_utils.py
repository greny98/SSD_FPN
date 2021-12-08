"""
Chuyển đổi tọa độ của box:
    - Swap x,y
    - xmin, ymin, xmax, ymax <=> cx, cy, w, h
    - IoU
"""

import tensorflow as tf


def swap_xy(boxes):
    """
    Chuyển đổi tọa độ của x và y
    :param boxes: shape=(n_boxes,4)
    :return:
    """
    return tf.stack([boxes[..., 1], boxes[..., 0], boxes[..., 3], boxes[..., 2]])


def corner_to_center(boxes):
    """
    Chuyển đổi tọa độ (xmin, ymin, xmax, ymax) => (cx, cy, w, h)
    :param boxes: (n_boxes, 4)
    :return:
    """
    center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    size = boxes[..., 2:] - boxes[..., :2]
    return tf.concat([center, size], axis=-1)


def center_to_corner(boxes):
    """
    Chuyển đổi tọa độ (cx, cy, w, h) => (xmin, ymin, xmax, ymax)
    :param boxes:
    :return:
    """
    top_left = boxes[..., :2] - 0.5 * boxes[..., 2:]
    bot_right = boxes[..., :2] + 0.5 * boxes[..., 2:]
    return tf.concat([top_left, bot_right], axis=-1)


def compute_iou(anchors: tf.Tensor, gt_boxes: tf.Tensor, mode='center'):
    if mode == 'center':
        anchors = center_to_corner(anchors)
        gt_boxes = center_to_corner(gt_boxes)
    anchors = tf.expand_dims(anchors, axis=1)
    inter_tl = tf.maximum(anchors[..., :2], gt_boxes[..., :2])
    inter_br = tf.minimum(anchors[..., 2:], gt_boxes[..., 2:])
    inter_area = inter_br - inter_tl
    inter_area = inter_area[..., 0] * inter_area[..., 1]
    anchors_area = anchors[..., 2:] - anchors[..., :2]
    anchors_area = anchors_area[..., 1] * anchors_area[..., 0]
    gt_area = gt_boxes[..., 2:] - gt_boxes[..., :2]
    gt_area = gt_area[..., 1] * gt_area[..., 0]
    union_area = anchors_area + gt_area - inter_area
    union_area = tf.maximum(union_area, 1e-8)
    return tf.clip_by_value(inter_area / union_area, 0., 1.)
