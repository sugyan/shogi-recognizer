import tensorflow as tf


def tfrecord_dataset(filepath):
    def parser(example):
        feature_description = {
            'image': tf.io.FixedLenFeature((), tf.string),
            'label': tf.io.FixedLenFeature((), tf.int64)
        }
        features = tf.io.parse_single_example(example, feature_description)
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, features['label']

    dataset = tf.data.TFRecordDataset(filepath)
    size = 0
    for _ in dataset:
        size += 1

    return dataset.map(parser), size
