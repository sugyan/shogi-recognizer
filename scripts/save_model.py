import tensorflow as tf
from nets.mobilenet import mobilenet_v2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_path', 'logdir/model.checkpoint',
                           '''Path to checkpoint file''')
tf.app.flags.DEFINE_string('labels', 'labels.txt',
                           '''Path to labels file''')
tf.app.flags.DEFINE_string("export_dir", 'output',
                           """Path to write the SavedModel""")


def main(argv=None):
    with tf.gfile.Open(FLAGS.labels) as f:
        labels = [line.strip() for line in f.readlines()]
    labels_str = tf.constant(list(','.join(labels).encode()), dtype=tf.int32, name='labels')

    placeholder = tf.placeholder(tf.float32, shape=(None, 96, 96, 3))
    logits, _ = mobilenet_v2.mobilenet(placeholder, len(labels))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.checkpoint_path)

        tf.saved_model.simple_save(sess,
                                   FLAGS.export_dir,
                                   inputs={'placeholder': placeholder},
                                   outputs={'labels': labels_str, 'output': logits})


if __name__ == '__main__':
    tf.app.run(main)
