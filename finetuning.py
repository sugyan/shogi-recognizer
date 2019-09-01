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
        return image, features['label']

    dataset = tf.data.TFRecordDataset(filepath).map(parser)
    size = 0
    for _ in dataset:
        size += 1
    return dataset.shuffle(size), size


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
    model.trainable = False

    # testing_data, testing_size = tfrecord_dataset(os.path.join(args.data_dir, 'testing.tfrecord'))
    # for images, labels in testing_data.batch(args.batch_size).take(3):
    #     print(tf.keras.backend.argmax(model(tf.image.convert_image_dtype(images, tf.float32))), labels)

    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])

    training_data, training_size = tfrecord_dataset(os.path.join(args.data_dir, 'training.tfrecord'))
    for images, labels in training_data.batch(training_size).take(1):
        generator = tf.keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=1,
            height_shift_range=1,
            rotation_range=1,
            brightness_range=(0.9, 1.1),
            zoom_range=0.01,
            rescale=1./255)
        training_datagen = generator.flow(
            images,
            tf.keras.utils.to_categorical(labels, 29),
            batch_size=args.batch_size)

    validation_data, validation_size = tfrecord_dataset(os.path.join(args.data_dir, 'validation.tfrecord'))
    for images, labels in validation_data.batch(validation_size).take(1):
        generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        validation_datagen = generator.flow(
            images,
            tf.keras.utils.to_categorical(labels, 29),
            batch_size=args.batch_size)

    history = model.fit_generator(
        training_datagen,
        epochs=10,
        steps_per_epoch=30,
        validation_data=validation_datagen)

    print(history.history)

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
    parser.add_argument(
        '--batch_size',
        help='''Batch size''',
        type=int,
        default=32)
    args = parser.parse_args()
    run(args)
