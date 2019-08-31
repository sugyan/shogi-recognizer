import argparse
import os
import tensorflow as tf

IMAGE_SIZE = (96, 96)


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

    return tf.data.TFRecordDataset(filepath).map(parser)


def run(args):
    with open(os.path.join(args.data_dir, 'labels.txt')) as fp:
        labels = [line.strip() for line in fp.readlines()]

    trained_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(None, 1280)),
        tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(
            len(labels), activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    ])
    trained_model.build((None, 1280))
    trained_model.load_weights(os.path.join(args.weights_dir, 'transfer.h5'))

    model = tf.keras.Sequential([
        tf.keras.applications.MobileNetV2(
            input_shape=IMAGE_SIZE + (3,),
            include_top=False,
            pooling='avg',
            weights='imagenet'),
        trained_model,
    ])
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    training_data = tfrecord_dataset(os.path.join(args.data_dir, 'training.tfrecord'))
    for images, labels in training_data.batch(32).take(3):
        print(tf.keras.backend.argmax(model(images)), labels)

    # model.fit(training_data)

    model.save(os.path.join(args.weights_dir, 'finetuning.h5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='''Path to directory of tfrecord files''',
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'data'))
    parser.add_argument(
        '--weights_dir',
        help='''Path to directory of weights files''',
        type=str,
        default='weights')
    args = parser.parse_args()
    run(args)
