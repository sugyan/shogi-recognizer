import tensorflow as tf
from tensorflow.python.framework import graph_util
from nets.mobilenet import mobilenet_v2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_path', 'logdir/model.checkpoint',
                           '''Path to checkpoint file''')
tf.app.flags.DEFINE_string('labels', 'labels.txt',
                           '''Path to labels file''')
tf.app.flags.DEFINE_string("output_graph", '',
                           """Path to write the frozen 'GraphDef'""")


def main(argv=None):
    with tf.gfile.Open(FLAGS.labels) as f:
        labels = [line.strip() for line in f.readlines()]

    placeholder = tf.placeholder(tf.float32, shape=(None, 96, 96, 3))
    logits, _ = mobilenet_v2.mobilenet(placeholder, len(labels))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.checkpoint_path)
        output = graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), ['MobilenetV2/Logits/output'])
    with open(FLAGS.output_graph, 'wb') as f:
        f.write(output.SerializeToString())


if __name__ == '__main__':
    tf.app.run(main)
