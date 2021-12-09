import albumentations as augment
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from configs.common_config import IMAGE_SIZE


def detect_augmentation(label_encoder, training):
    if training:
        transform = augment.Compose([
            augment.LongestMaxSize(320),
            augment.ImageCompression(quality_lower=75, quality_upper=100),
            augment.HorizontalFlip(p=0.3),
            # augment.VerticalFlip(p=0.3),
            # augment.RandomRotate90(p=0.3),
            augment.RandomBrightnessContrast(0.3, 0.3, p=0.3),
            # augment.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.01, rotate_limit=10, p=0.4),
            # augment.GaussNoise(p=0.3),
            # augment.GaussianBlur(p=0.4),
            augment.RandomSizedBBoxSafeCrop(IMAGE_SIZE, IMAGE_SIZE, p=0.5),
            augment.Resize(IMAGE_SIZE, IMAGE_SIZE),
        ], bbox_params=augment.BboxParams(format='pascal_voc'))
    else:
        transform = augment.Compose([augment.Resize(IMAGE_SIZE, IMAGE_SIZE)],
                                    bbox_params=augment.BboxParams(format='pascal_voc'))

    def augmentation(image, bboxes, labels):
        if bboxes.shape[0] == 0:
            print("\n===========")
            print("Empty BBox!!!")
            print("===========\n")
        img_h, img_w, _ = image.shape
        bboxes = np.clip(bboxes, 0., 1.0)
        # Hande bboxes
        ymin = bboxes[:, [0]] * img_h
        xmin = bboxes[:, [1]] * img_w
        ymax = bboxes[:, [2]] * img_h
        xmax = bboxes[:, [3]] * img_w

        extend_boxes = np.ones(shape=(bboxes.shape[0], 1))
        bboxes = np.concatenate([xmin, ymin, xmax, ymax, extend_boxes], axis=-1)
        mask = (bboxes[:, 3] > bboxes[:, 1]) & (bboxes[:, 2] > bboxes[:, 0])
        bboxes = bboxes[mask]
        labels = labels[mask]
        # Add info
        data = {'image': image, 'bboxes': bboxes}
        transformed = transform(**data)
        # extract transformed image
        aug_img = transformed['image']
        aug_img = tf.cast(aug_img, tf.float32)
        aug_img = aug_img / 127.5 - 1
        # extract boxes
        bboxes_transformed = []
        for x, y, w, h, _ in transformed['bboxes']:
            cx = x + 0.5 * w
            cy = y + 0.5 * w
            bboxes_transformed.append(tf.convert_to_tensor([cx, cy, w, h], tf.float32))

        bboxes_transformed = tf.convert_to_tensor(bboxes_transformed, tf.float32)
        labels = tf.convert_to_tensor(labels, tf.float32)
        labels = label_encoder.encode_sample(aug_img.shape, bboxes_transformed, labels)

        return [aug_img, labels]

    return augmentation


def create_coco(label_encoder, batch_size):
    def preprocess_image(info, training):
        image = info["image"]
        bboxes = info["objects"]["bbox"]
        labels = info["objects"]["label"]
        aug_image, labels = tf.numpy_function(func=detect_augmentation(label_encoder, training),
                                              inp=[image, bboxes, labels],
                                              Tout=[tf.float32, tf.float32])
        return aug_image, labels

    def check_have_bboxes(bboxes):
        return bboxes.shape[0] > 0

    def filter_ds(info):
        bboxes = info["objects"]["bbox"]
        valid = tf.numpy_function(
            func=check_have_bboxes,
            inp=[bboxes],
            Tout=[tf.bool])
        return info, valid

    (train_ds, val_ds), info = tfds.load(
        'coco/2017', split=['train', 'validation'], with_info=True, shuffle_files=True)

    train_ds = train_ds.repeat()
    train_ds = (train_ds.map(filter_ds, num_parallel_calls=tf.data.AUTOTUNE)
                .filter(lambda info, valid: valid)
                .map(lambda info, valid: info))

    val_ds = val_ds.repeat()
    val_ds = (val_ds.map(filter_ds, num_parallel_calls=tf.data.AUTOTUNE)
              .filter(lambda info, valid: valid)
              .map(lambda info, valid: info))
    train_ds = (train_ds.shuffle(512, reshuffle_each_iteration=True)
                .map(lambda info: preprocess_image(info, True), num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = (val_ds.shuffle(512).map(
        lambda info: preprocess_image(info, False),
        num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE))
    return train_ds, val_ds
