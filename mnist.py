'''
A Reccurent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Example code is adapted from https://github.com/aymericdamien/TensorFlow-Examples/
Author: Parminder
'''

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
from layers import *
# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a reccurent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("iterations", 100000,
                          "Number of iterations.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("display_step", 10,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("hidden", 128,
                            "How many hidden units.")
tf.app.flags.DEFINE_integer("classes", 10,
                            "NUmber of classes")
tf.app.flags.DEFINE_integer("layers", 1,
                            "NUmber of layers for the model")
tf.app.flags.DEFINE_string("cell_type", "LNGRU" , "Select from LSTM, GRU , BasicRNN, LNGRU, LNLSTM")
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("summaries_dir", "./log/" , "Directory for summary")
FLAGS = tf.app.flags.FLAGS
# Parameters
learning_rate = FLAGS.learning_rate
training_iters = FLAGS.iterations
batch_size = FLAGS.batch_size
display_step = FLAGS.display_step

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = FLAGS.hidden # hidden layer num of features
n_classes = FLAGS.classes # MNIST total classes (0-9 digits)


def train():
    sess = tf.InteractiveSession()


    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, n_steps, n_input], name='x-input')
        y = tf.placeholder(tf.float32, [None, n_classes], name='y-input')

    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }



    def RNN(x, weights, biases, type):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, n_steps, x)
        # Define a lstm cell with tensorflow
        cell_class_map = {
             "LSTM": rnn_cell.BasicLSTMCell(n_hidden),
             "GRU": rnn_cell.GRUCell(n_hidden),
             "BasicRNN": rnn_cell.BasicRNNCell(n_hidden),
             "LNGRU": LNGRUCell(n_hidden),
             "LNLSTM": LNBasicLSTMCell(n_hidden)}

        lstm_cell = cell_class_map.get(type)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * FLAGS.layers)
        print "Using %s model" % type
        # Get lstm cell output
        outputs, states = rnn.rnn(cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']


    pred = RNN(x, weights, biases, FLAGS.cell_type)

    # Define loss and optimizer
    # print pred
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.scalar_summary('Accuracy', accuracy)
    tf.scalar_summary('Cost', cost)

    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + "train/",
                                              sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + "test/",
                                              sess.graph)
    # Initializing the variables
    init = tf.initialize_all_variables()

    sess.run(init)
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # Run optimization op (backprop)
            summary, _ = sess.run([merged,optimizer], feed_dict={x: batch_x, y: batch_y})
            # train_writer.add_summary(summary, step)
            if step % display_step == 0:
                # Calculate batch accuracy
                summary, acc, loss = sess.run([merged,accuracy,cost], feed_dict={x: batch_x, y: batch_y})
                train_writer.add_summary(summary, step)
                # Calculate batch loss
                print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc)

                summary, acc, loss = sess.run([merged, accuracy, cost], feed_dict={x: test_data, y: test_label})
                test_writer.add_summary(summary, step)
                print "Testing Accuracy:", acc
            step += 1
    print "Optimization Finished!"

    # Calculate accuracy for 128 mnist test images

    print "Testing Accuracy:", \
         sess.run(accuracy, feed_dict={x: test_data, y: test_label})


def main(_):
  train()


if __name__ == '__main__':
  tf.app.run()
