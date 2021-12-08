import tensorflow as tf
from tensorflow.keras import losses


class ClassificationLoss(losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.):
        super(ClassificationLoss, self).__init__(reduction='none', name="ClassificationLoss")
        self._gamma = gamma
        self._alpha = alpha

    def call(self, y_true, y_pred):
        # Calculate
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)


class LocalizationLoss(losses.Loss):
    def __init__(self, delta=1.):
        super(LocalizationLoss, self).__init__(reduction="none", name="LocalizationLoss")
        self._delta = delta

    def call(self, y_true, y_pred):
        # Tính loss của positive boxes và top các negative có loss cao
        diff = y_true - y_pred
        abs_diff = tf.abs(diff)
        square_diff = tf.square(diff)
        loss = tf.where(tf.less(abs_diff, self._delta), 0.5 * square_diff, abs_diff - 0.5)
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetLoss(losses.Loss):
    def __init__(self, num_classes=4, alpha=0.25, gamma=2., delta=1):
        super(RetinaNetLoss, self).__init__(reduction='auto', name='RetinaNetLoss')
        self._num_classes = num_classes
        self.focal_loss = ClassificationLoss(alpha=alpha, gamma=gamma)
        self.l1_smooth_loss = LocalizationLoss(delta=delta)

    def call(self, y_true, y_pred):
        print(y_true, y_pred)
        y_pred = tf.cast(y_pred, tf.float32)
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]
        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[:, :, 4:]
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
        clf_loss = self.focal_loss(cls_labels, cls_predictions)
        box_loss = self.l1_smooth_loss(box_labels, box_predictions)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = clf_loss + box_loss
        return loss
