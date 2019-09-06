import argparse
import glob
import os
import tensorflow as tf

from dataset import tfrecord_dataset
from model import mobilenet_v2


def dump_features(data_dir, features_dir):
    model = mobilenet_v2()
    model.trainable = False
    model.summary()

    for filepath in glob.glob(os.path.join(data_dir, '*.tfrecord')):
        category = os.path.splitext(os.path.basename(filepath))[0]
        print(f'calculate features of {category} data', end='', flush=True)
        dataset, _ = tfrecord_dataset(filepath)
        with tf.io.TFRecordWriter(os.path.join(features_dir, f'{category}.tfrecord')) as writer:
            for i, (image, label) in enumerate(dataset):
                features = model.predict(tf.expand_dims(image, axis=0))
                feature = {
                    'features': tf.train.Feature(
                        float_list=tf.train.FloatList(value=features.flatten().tolist())),
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                if i % 50 == 0:
                    print('.', end='', flush=True)
        print()


def run(args):
    if len(glob.glob(os.path.join(args.features_dir, '*.tfrecord'))) == 0:
        os.makedirs(args.features_dir, exist_ok=True)
        dump_features(args.data_dir, args.features_dir)

    def dataset(category):
        def parse_example(serialized):
            features = {
                'features': tf.io.FixedLenFeature([1280], tf.float32),
                'label': tf.io.FixedLenFeature([], tf.int64),
            }
            example = tf.io.parse_single_example(serialized, features)
            return example['features'], example['label']

        ds = tf.data.TFRecordDataset(os.path.join(args.features_dir, f'{category}.tfrecord'))
        size = 0
        for _ in ds:
            size += 1
        return ds.map(parse_example), size

    training_data, training_size = dataset('training')
    validation_data, validation_size = dataset('validation')

    with open(os.path.join(args.data_dir, 'labels.txt')) as fp:
        labels = [line.strip() for line in fp.readlines()]
    classes = len(labels)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((1280,)),
        tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(
            classes,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    ])
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    history = model.fit(
        training_data.shuffle(training_size).repeat().batch(args.batch_size),
        steps_per_epoch=training_size // args.batch_size,
        epochs=30,
        validation_data=validation_data.shuffle(validation_size).batch(args.batch_size),
        validation_steps=validation_size // args.batch_size,
        callbacks=[tf.keras.callbacks.TensorBoard()])
    print(history.history)

    testing_data, testing_size = dataset('testing')
    test_result = model.evaluate(
        testing_data.shuffle(testing_size).batch(args.batch_size),
        steps=testing_size // args.batch_size,
        verbose=0)
    print(test_result)

    model.save_weights(os.path.join(args.weights_dir, 'transfer.h5'))
    classifier = tf.keras.Sequential([
        mobilenet_v2(),
        model,
    ])
    classifier.trainable = False
    classifier.save('transfer.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='''Path to directory of tfrecord files''',
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'data'))
    parser.add_argument(
        '--features_dir',
        help='''Path to directory of features files''',
        type=str,
        default='features')
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
