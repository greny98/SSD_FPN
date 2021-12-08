import tensorflow as tf
from keras.models import Model

from configs.common_config import IMAGE_SIZE
from model.anchor_boxes import PredictionDecoder
from model.ssd import create_ssd_model


def PredictModel(num_classes, weights=None):
    ssd_model = create_ssd_model(num_classes)
    if weights is not None:
        ssd_model.load_weights(weights)
    image = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3,), name="image")
    predictions = ssd_model(image, training=False)
    detections = PredictionDecoder()(image, predictions)
    inference_model = Model(inputs=image, outputs=detections)
    return inference_model
