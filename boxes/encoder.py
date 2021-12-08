import tensorflow as tf

from boxes import boxes_utils
from boxes.anchor_boxes import AnchorBoxes


class LabelEncoder:
    def __init__(self):
        self._anchor_boxes = AnchorBoxes()
        self._box_variance = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2], tf.float32)

    @staticmethod
    def _match_anchor_boxes(anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4):
        iou_matrix = boxes_utils.compute_iou(anchor_boxes, gt_boxes, mode='center')
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:])
            ],
            axis=-1
        )
        box_target = box_target / self._box_variance
        return box_target

    def encode_sample(self, image_shape, gt_boxes, cls_ids):
        anchor_boxes = self._anchor_boxes.create_anchors_boxes(image_shape[1], image_shape[0])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(anchor_boxes, gt_boxes)
        # Compute boxes
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        # Compute cls
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(tf.equal(positive_mask, 1.0), matched_gt_cls_ids, -1.0)
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label
