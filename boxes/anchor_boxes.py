import tensorflow as tf


class AnchorBoxes:
    def __init__(self):
        self._aspect_ratios = [0.5, 1.0, 2.0]
        self._scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]
        self._num_anchors = len(self._aspect_ratios) * len(self._scales)
        # step = int((IMAGE_SIZE - 32) / 4)
        self._areas = [(x * 36 + 16) ** 2 for x in range(5)]
        self._strides = [2 ** i for i in range(3, 8)]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self._aspect_ratios:
                # Tính width và height ứng với mỗi area
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1),
                    shape=[1, 1, 2]
                )
                for scale in self._scales:
                    # Tính cho mỗi scale
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_width, feature_height, level):
        # level bắt đầu từ 3
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(self._anchor_dims[level - 3], [feature_width, feature_height, 1, 1])
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(anchors, shape=(-1, 4))

    def create_anchors_boxes(self, image_width, image_height):
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_width / 2 ** i),
                tf.math.ceil(image_height / 2 ** i),
                i)
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)
