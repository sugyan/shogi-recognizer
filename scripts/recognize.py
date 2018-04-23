import numpy as np
import tensorflow as tf
from PIL import Image

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('graph', 'output_graph.pb',
                           '''Path to output graph file''')
tf.app.flags.DEFINE_string('labels', 'output_labels.txt',
                           '''Path to output labels file''')
tf.app.flags.DEFINE_string('input_image', '',
                           '''Path to input image file''')
IMAGE_SIZE = 96


class Recognizer:
    def __init__(self, graph, labels):
        self.graph_def = tf.GraphDef()
        with tf.gfile.Open(graph, 'rb') as f:
            self.graph_def.ParseFromString(f.read())
        with tf.gfile.Open(labels) as f:
            self.labels = [line.strip() for line in f.readlines()]

    def run(self, input_image):
        img = Image.open(input_image).convert('RGB')
        h, w = img.height / 9, img.width / 9
        size = max([h, w])
        inputs = []
        for file in range(1, 10):
            for rank in range(1, 10):
                cropped = img.crop([
                    w * (9.5 - file) - size / 2.0,
                    h * (rank - 0.5) - size / 2.0,
                    w * (9.5 - file) + size / 2.0,
                    h * (rank - 0.5) + size / 2.0])
                resized = cropped.resize([IMAGE_SIZE, IMAGE_SIZE])
                if file == 2 and rank == 8:
                    with open('out.jpg', 'w') as f:
                        resized.save(f)
                inputs.append(np.array(resized) * 2.0 / 255.0 - 1.0)
        with tf.Graph().as_default() as g:
            tf.import_graph_def(self.graph_def, name='')
            placeholder = g.get_tensor_by_name('Placeholder:0')
            final_result = g.get_tensor_by_name('final_result:0')
            top3 = tf.nn.top_k(final_result, 3)
            with tf.Session() as sess:
                results, indices = sess.run(top3, feed_dict={placeholder: inputs})
        # for result in results:
        #     print(result)
        # board = [[]]
        for rank in range(9):
            row = ''
            for file in range(9):
                i = (8 - file) * 9 + rank
                best = indices[i][0]
                label = self.labels[best]
                s = label.split(' ')
                if len(s) == 1:
                    if label == 'other':
                        row += ' ? '
                    if label == 'blank':
                        row += ' * '
                else:
                    if s[0] == 'b':
                        row += '+'
                    if s[0] == 'w':
                        row += '-'
                    row += s[1].upper()
            print(row)


if __name__ == '__main__':
    Recognizer(FLAGS.graph, FLAGS.labels).run(FLAGS.input_image)
