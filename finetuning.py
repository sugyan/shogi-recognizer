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

    dataset = tf.data.TFRecordDataset(filepath).map(parser)
    size = 0
    for _ in dataset:
        size += 1
    return dataset.shuffle(size), size


def build_model(classes):
    return tf.keras.Sequential([
        tf.keras.applications.MobileNetV2(
            input_shape=IMAGE_SIZE + (3,),
            include_top=False,
            pooling='avg',
            weights='imagenet'),
        tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(
            classes,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    ])


def train(data_dir, weights_dir, batch_size):
    with open(os.path.join(data_dir, 'labels.txt')) as fp:
        labels = [line.strip() for line in fp.readlines()]

    model = build_model(len(labels))
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    training_data, training_size = tfrecord_dataset(os.path.join(data_dir, 'training.tfrecord'))
    for images, labels in training_data.batch(training_size).take(1):
        generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=2,
            width_shift_range=2,
            height_shift_range=2,
            brightness_range=(0.9, 1.1),
            channel_shift_range=0.1,
            zoom_range=0.01,
            rescale=1./255)
        training_datagen = generator.flow(images, labels, batch_size=batch_size)

    validation_data, validation_size = tfrecord_dataset(os.path.join(data_dir, 'validation.tfrecord'))
    testing_data, testing_size = tfrecord_dataset(os.path.join(data_dir, 'testing.tfrecord'))

    history = model.fit_generator(
        training_datagen,
        epochs=50,
        steps_per_epoch=training_size // batch_size,
        validation_steps=validation_size // batch_size,
        validation_data=validation_data.batch(batch_size),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(weights_dir, 'weights.{epoch:02d}-{val_loss:.5f}.h5'),
                save_weights_only=True),
        ])
    print(history.history)

    test_result = model.evaluate(testing_data.batch(batch_size), steps=testing_size // batch_size)
    print(test_result)

    model.save(os.path.join(weights_dir, 'finetuning.h5'))


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
    parser.add_argument(
        '--batch_size',
        help='''Batch size''',
        type=int,
        default=32)
    args = parser.parse_args()
    train(args.data_dir, args.weights_dir, args.batch_size)
