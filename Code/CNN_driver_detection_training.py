# edited by SJLIM, DWJEONG, MSKIM
import os
import sys
import numpy
import tensorflow as tf
from tensorflow.python.platform import gfile
import ExtractData_train_validation as extractData_oned_train_val


# user configuration
SAVE_dir = 'train_dir'
# DATA_TYPES = 'prepro'
DRIVERS = 'CS,HH,JH,TW'
LABELS = '0,1,2,3'


# model parameter
TRAIN_SIZE =        50000
VALIDATION_SIZE =   20000

BATCH_SIZE = 100
NUM_EPOCHS = 5

NUM_ROWS =  300
DATA_SIZE = 2
NUM_CHANNELS = DATA_SIZE

CONV_1_W = 1
CONV_1_H = 5
CONV_1_PADDING = 'SAME'

CONV_2_W = 1
CONV_2_H = 4
CONV_2_PADDING = 'SAME'

CONV_3_W = 1
CONV_3_H = 4
CONV_3_PADDING = 'SAME'

CONV_1_D =  32
CONV_2_D =  64
CONV_3_D =  128
FULL_D =    512

POOL_1 = 2
POOL_2 = 2
POOL_3 = 2

SEED = 66478  # Set to None for random seed.

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")

#user inputs.
tf.app.flags.DEFINE_string('train_dir', './%s' %(SAVE_dir), """Directory where to write event logs """
                           """and checkpoint.""")
# tf.app.flags.DEFINE_string('data_types', DATA_TYPES, "Set the data types")
tf.app.flags.DEFINE_string('drivers', DRIVERS, "Set the drivers")
tf.app.flags.DEFINE_string('labels', LABELS, "Set the labels")

tf.app.flags.DEFINE_integer('train_size',   TRAIN_SIZE, "Set")
tf.app.flags.DEFINE_integer('val_size',     VALIDATION_SIZE, "Set")

tf.app.flags.DEFINE_integer('batch_size', BATCH_SIZE, "Set")
tf.app.flags.DEFINE_integer('num_epochs', NUM_EPOCHS, "Set")

tf.app.flags.DEFINE_integer('num_rows', NUM_ROWS, "Set")
tf.app.flags.DEFINE_integer('data_size',DATA_SIZE, "Set")

tf.app.flags.DEFINE_integer('c1w', CONV_1_W, "Set")
tf.app.flags.DEFINE_integer('c1h', CONV_1_H, "Set")

tf.app.flags.DEFINE_integer('c2w', CONV_2_W, "Set")
tf.app.flags.DEFINE_integer('c2h', CONV_2_H, "Set")

tf.app.flags.DEFINE_integer('c3w', CONV_3_W, "Set")
tf.app.flags.DEFINE_integer('c3h', CONV_3_H, "Set")

tf.app.flags.DEFINE_integer('c1d', CONV_1_D, "Set")
tf.app.flags.DEFINE_integer('c2d', CONV_2_D, "Set")
tf.app.flags.DEFINE_integer('c3d', CONV_3_D, "Set")
tf.app.flags.DEFINE_integer('fd', FULL_D, "Set")

tf.app.flags.DEFINE_integer('p1', POOL_1, "Set")
tf.app.flags.DEFINE_integer('p2', POOL_2, "Set")
tf.app.flags.DEFINE_integer('p3', POOL_3, "Set")
FLAGS = tf.app.flags.FLAGS


# model parameter
SAVE_dir = FLAGS.train_dir

TRAIN_SIZE =        FLAGS.train_size
VALIDATION_SIZE =   FLAGS.val_size

BATCH_SIZE = FLAGS.batch_size
NUM_EPOCHS = FLAGS.num_epochs

NUM_ROWS =  FLAGS.num_rows
DATA_SIZE = FLAGS.data_size

CONV_1_W = FLAGS.c1w
CONV_1_H = FLAGS.c1h

CONV_2_W = FLAGS.c2w
CONV_2_H = FLAGS.c2h

CONV_3_W = FLAGS.c3w
CONV_3_H = FLAGS.c3h

CONV_1_D =  FLAGS.c1d
CONV_2_D =  FLAGS.c2d
CONV_3_D =  FLAGS.c3d
FULL_D =    FLAGS.fd

POOL_1 = FLAGS.p1
POOL_2 = FLAGS.p2
POOL_3 = FLAGS.p3


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    print 'predictions'
    print numpy.shape(predictions)
    print 'labels'
    print numpy.shape(labels)
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

# 'Xavier_init' is method of weight initalization.
def xavier_init(n_inputs, n_outputs, uniform=False):
    if uniform:
        # 6 was used in the paper.
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

def xavier_init2(n_inputs, n_outputs2,n_outputs3, uniform=False):
    if uniform:
        # 6 was used in the paper.
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs2*n_outputs3))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean.
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs2*n_outputs3))
        return tf.truncated_normal_initializer(stddev=stddev)


# Process of Convolutional neural network.
def main(argv=None):
    if FLAGS.self_test:
        num_epochs = 1
    else:
        # Extract it into numpy arrays.
        DRIVERS = FLAGS.drivers.split(',')
        # DATA_TYPES = FLAGS.data_types.split(',')
        LABELS = FLAGS.labels.split(',')
        print 'drivers', DRIVERS
        # print 'data_types', DATA_TYPES
        print 'labels', LABELS
        NUM_LABELS = len(set(LABELS))
        print 'num_labels', NUM_LABELS

        # Extract data and Using 80% of all data as training data.
        train_data, train_labels = extractData_oned_train_val.extract_data_oned(numRows = NUM_ROWS, numData = TRAIN_SIZE,
                                                                                drivers = DRIVERS, labels = LABELS, mode = 'train', DATA_SIZE = DATA_SIZE,
                                                                                NUM_CHANNELS=NUM_CHANNELS,ONED=True)
        train_data = train_data[:, :, :]
        print "train_data", numpy.shape(train_data)

        # Extract data and Using 20% of all data as validation data.
        validation_data, validation_labels = extractData_oned_train_val.extract_data_oned(numRows = NUM_ROWS, numData = VALIDATION_SIZE,
                                                                                          drivers = DRIVERS, labels = LABELS, mode = 'validate',DATA_SIZE = DATA_SIZE,
                                                                                          NUM_CHANNELS=NUM_CHANNELS, ONED=True)
        validation_data = validation_data[:, :, :]
        print "validation_data", numpy.shape(validation_data)

        # Generate a validation set.
        num_epochs = NUM_EPOCHS

    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)

    train_size = train_labels.shape[0]

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, NUM_ROWS,  NUM_CHANNELS))
    print "train_data_node", train_data_node
    print "train_data_node", train_data_node[0]
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))

    # For the validation and test data, we'll just hold the entire dataset in one constant node.
    print "validation_data", numpy.shape(validation_data)
    validation_data_node = tf.constant(validation_data)
    print "validation_data_node", validation_data_node

    # Weights and biases of convolutional layer 1.
    # Filter size of convolutional layer 1 : 1 x 5.
    # Filter number of convolutional layer 1 : 32.
    conv1_weights = tf.get_variable("C1", shape=[CONV_1_H, NUM_CHANNELS, CONV_1_D], initializer=xavier_init2(CONV_1_H, NUM_CHANNELS, CONV_1_D))
    conv1_biases = tf.Variable(tf.zeros([CONV_1_D]))

    # Weights and biases of convolutional layer 2.
    # Filter size of convolutional layer 2 : 1 x 4.
    # Filter number of convolutional layer 2 : 64.
    conv2_weights = tf.get_variable("C2", shape=[CONV_2_H, CONV_1_D, CONV_2_D], initializer=xavier_init2(CONV_2_H, CONV_1_D, CONV_2_D))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[CONV_2_D]))

    # Weights and biases of convolutional layer 3.
    # Filter size of convolutional layer 3 : 1 x 4.
    # Filter number of convolutional layer 3 : 128.
    conv3_weights = tf.get_variable("C3", shape=[CONV_3_H, CONV_2_D, CONV_3_D], initializer=xavier_init2(CONV_3_H, CONV_2_D, CONV_3_D))
    conv3_biases = tf.Variable(tf.constant(0.1, shape=[CONV_3_D]))

    # Number of trained weight.
    CONV1H = CONV_1_H
    CONV2H = CONV_2_H
    if CONV_1_PADDING == 'SAME':
        CONV1H = 1
    if CONV_2_PADDING == 'SAME':
        CONV2H = 1
    WH = ((NUM_ROWS - CONV1H + 1) / (POOL_1) -CONV2H+ 1) / POOL_2 * CONV_2_D + 64
    print 'WH', WH

    # Classify the trained weight throught convolutional filter
    # Fully connected layer 1.
    # Number of Fully connected layer 1 : 512
    fc1_weights = tf.get_variable("W1", shape=[WH, FULL_D], initializer=xavier_init(WH, FULL_D))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[FULL_D]))

    # Fully connected layer 2.
    # Number of Fully connected layer 2 : 4 (number of drivers)
    fc2_weights = tf.get_variable("W2", shape=[FULL_D, NUM_LABELS], initializer=xavier_init(FULL_D, NUM_LABELS))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))



    def model(data, train):
        """The Model definition."""
        # 1D convolution, with 'SAME' padding (i.e. the output feature map has the same size as the input).
        # Method of pool is used Maxpooling.
        # Pooling and relu of convolutional layer 1.
        # Maxpooling size : 2.
        conv = tf.nn.conv1d(data, conv1_weights, stride=1, padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        print relu
        # Max pooling. The kernel size spec {ksize} also follows the layout of the data.
        # Here we have a pooling window of 2, and a stride of 2.
        if True:
            pool = tf.layers.max_pooling1d(relu,POOL_1,strides=POOL_1, padding='SAME')
            print pool
        else:
            pool = relu
            print pool

        # Pooling and relu of convolutional layer 2.
        # Maxpooling size : 2.
        conv = tf.nn.conv1d(pool, conv2_weights, stride=1, padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        if True:
            pool = tf.layers.max_pooling1d(relu, POOL_2, strides=POOL_2,padding='SAME')
            print pool
        else:
            pool = relu
            print pool

        # Pooling and relu of convolutional layer 3.
        # Maxpooling size : 2.
        conv = tf.nn.conv1d(pool, conv3_weights, stride=1, padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
        if True:
            pool = tf.layers.max_pooling1d(relu, POOL_3, strides=POOL_3,padding='SAME')
        else:
            pool = relu
            print pool


        # Reshape the feature map cuboid into a 1D matrix to feed it to the fully connected layers.
        pool_shape = pool.get_shape().as_list()
        print pool_shape
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2]])

        # Fully connected layer. Note that the '+' operation automatically broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        print "hidden", hidden
        print 'before train work'
        if train:
            # Add a 50% dropout during training only. Dropout also scales
            # activations such that no rescaling is needed at evaluation time.
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
            print 'train work'
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    # Training computation : logits + cross-entropy loss.
    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=train_labels_node))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
    batch = tf.Variable(0)
    saver = tf.train.Saver(tf.all_variables())

    # Decay once per epoch, using an exponential schedule starting at 0.001.
    learning_rate = tf.train.exponential_decay(
        0.001,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    # Use Adams optimization.
    optimizer = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
    train_prediction = tf.nn.softmax(logits)
    validation_prediction = tf.nn.softmax(model(validation_data_node,False))


    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        print 'Initialized!'

        # Loop through training steps.
        for step in xrange(int(num_epochs * train_size / BATCH_SIZE)):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), :, :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]

            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph is should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}

            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = s.run(
                [optimizer, loss, learning_rate, train_prediction],
                feed_dict=feed_dict)
            predictions = s.run(train_prediction,
                feed_dict=feed_dict)

            if step % 100 == 0:
                print 'Epoch %.2f' \
                      '' % (float(step) * BATCH_SIZE / train_size)
                print 'Minibatch loss: %.3f, learning rate: %.6f' % (l, lr)
                print 'Minibatch error: %.1f%%' % error_rate(predictions, batch_labels)
                print 'Validation error: %.1f%%' % error_rate(validation_prediction.eval(), validation_labels)

                sys.stdout.flush()
                checkpoint_path = os.path.join(FLAGS.train_dir, 'driver.ckpt')

        # Finally print the result!
        saver.save(s, checkpoint_path, global_step=step)
        print("save directory: %s" %(SAVE_dir))



if __name__ == '__main__':
    tf.app.run()