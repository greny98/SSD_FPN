"""
Decode boxes dựa vào kết quả predict
"""
import tensorflow as tf
from tensorflow.keras import layers

from boxes import boxes_utils
from boxes.anchor_boxes import AnchorBoxes


class PredictionDecoder(layers.Layer):
    def __init__(self, **kwargs):
        super(PredictionDecoder, self).__init__(**kwargs)
        self._box_variance = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2])
        self._anchor_boxes = AnchorBoxes()

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = boxes_utils.center_to_corner(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_boxes.create_anchors_boxes(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)
        return boxes, cls_predictions
