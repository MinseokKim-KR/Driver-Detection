# edited by SJLIM, DWJEONG, MSKIM.
from collections import Counter
import numpy
import numpy as np
import tensorflow as tf
import ExtractData_detecting as extractData
import matplotlib.pyplot as plt

# user configuration
# DATA_TYPES =      'prepro'
LABELS =          '0,1,2,3'
DRIVERS_NAME =    ['CS', 'HH', 'JH', 'TW']
PATH =            "train_dir"

DRIVER =          'CS'
LABEL_NUM =        0
Shifting_strides = 2


# model parameter
BATCH_SIZE = 100
NUM_EPOCHS = 5

NUM_ROWS =  300
DATA_SIZE = 2
NUM_CHANNELS=DATA_SIZE

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


# user inputs
tf.app.flags.DEFINE_string('restore_path', PATH, "Set")

# tf.app.flags.DEFINE_string('data_types', DATA_TYPES, "Set the data type")
tf.app.flags.DEFINE_string('driver', DRIVER, "Set the driver")
tf.app.flags.DEFINE_integer('label_num', LABEL_NUM, "Set the labels_num")
tf.app.flags.DEFINE_string('labels', LABELS, "Set the labels")
tf.app.flags.DEFINE_integer('shifting_strides', Shifting_strides, "Set")


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
PATH = FLAGS.restore_path
LABEL_NUM = FLAGS.label_num
Shifting_strides = FLAGS.shifting_strides


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
def main(argv=None):  # pylint: disable=unused-argument
    if FLAGS.self_test:
        num_epochs = 1
    else:
        # Extract it into numpy arrays.
        DRIVER = FLAGS.driver.split(',')
        # DATA_TYPES = FLAGS.data_types.split(',')
        LABELS = FLAGS.labels.split(',')
        print 'driver', DRIVER
        # print 'data_type', DATA_TYPES
        # print 'labels', LABELS
        NUM_LABELS = len(set(LABELS))

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:

    # Weights and biases of convolutional layer 1.
    # Filter size of convolutional layer 1 : 1 x 5.
    # Filter number of convolutional layer 1 : 32.
    conv1_weights = tf.get_variable("C1", shape=[CONV_1_H, NUM_CHANNELS, CONV_1_D],
                                    initializer=xavier_init2(CONV_1_H, NUM_CHANNELS, CONV_1_D))
    conv1_biases = tf.Variable(tf.zeros([CONV_1_D]))

    # Weights and biases of convolutional layer 2.
    # Filter size of convolutional layer 1 : 1 x 4.
    # Filter number of convolutional layer 1 : 64.
    conv2_weights = tf.get_variable("C2", shape=[CONV_2_H, CONV_1_D, CONV_2_D],
                                    initializer=xavier_init2(CONV_2_H, CONV_1_D, CONV_2_D))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[CONV_2_D]))

    # Weights and biases of convolutional layer 3.
    # Filter size of convolutional layer 1 : 1 x 4.
    # Filter number of convolutional layer 1 : 128.
    conv3_weights = tf.get_variable("C3", shape=[CONV_3_H, CONV_2_D, CONV_3_D],
                                    initializer=xavier_init2(CONV_3_H, CONV_2_D, CONV_3_D))
    conv3_biases = tf.Variable(tf.constant(0.1, shape=[CONV_3_D]))

    # Number of trained weight.
    CONV1H = CONV_1_H
    CONV2H = CONV_2_H
    if CONV_1_PADDING == 'SAME':
        CONV1H = 1
    if CONV_2_PADDING == 'SAME':
        CONV2H = 1
    WH = ((NUM_ROWS - CONV1H + 1) / (POOL_1) -CONV2H+ 1) / POOL_2 * CONV_2_D + 64
    # print WH

    # Classify the trained weight throught convolutional filter
    # Fully connected layer 1.
    # Number of Fully connected layer 1 : 512
    fc1_weights = tf.get_variable("W1", shape=[WH, FULL_D], initializer=xavier_init(WH, FULL_D))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[FULL_D]))

    # Fully connected layer 2.
    # Number of Fully connected layer 2 : 4 (number of drivers)
    fc2_weights = tf.get_variable("W2", shape=[FULL_D, NUM_LABELS], initializer=xavier_init(FULL_D, NUM_LABELS))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))


    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train):
        """The Model definition."""
        # 1D convolution, with 'SAME' padding (i.e. the output feature map has the same size as the input).
        # Method of pool is used Maxpooling.
        # Pooling and relu of convolutional layer 1.
        # Maxpooling size : 2.
        conv = tf.nn.conv1d(data, conv1_weights, stride=1, padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of the data.
        # Here we have a pooling window of 2, and a stride of 2.
        if True:
            pool = tf.layers.max_pooling1d(relu, POOL_1, strides=POOL_1, padding='SAME')
        else:
            pool = relu

        # Pooling and relu of convolutional layer 2.
        # Maxpooling size : 2.
        conv = tf.nn.conv1d(pool, conv2_weights, stride=1, padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        if True:
            pool = tf.layers.max_pooling1d(relu, POOL_2, strides=POOL_2, padding='SAME')
        else:
            pool = relu

        # Pooling and relu of convolutional layer 3.
        # Maxpooling size : 2.
        conv = tf.nn.conv1d(pool, conv3_weights, stride=1, padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
        if True:
            pool = tf.layers.max_pooling1d(relu, POOL_3, strides=POOL_3, padding='SAME')
        else:
            pool = relu

        # Reshape the feature map cuboid into a 1D matrix to feed it to the fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2]])

        # Fully connected layer. Note that the '+' operation automatically broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        if train:
            # Add a 50% dropout during training only. Dropout also scales
            # activations such that no rescaling is needed at evaluation time.
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    test_label = DRIVER
    saver = tf.train.Saver(tf.all_variables())
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        saver.restore(s, PATH + '/'+ 'driver.ckpt-2499')

        # Extract data of driver detection.
        test_data_list = []
        test_data = extractData.extract_data_oned(numRows=NUM_ROWS, numData=None, drivers=DRIVER, labels=LABELS, mode='test',
                                                               DATA_SIZE=DATA_SIZE, Shifting_strides=Shifting_strides, NUM_CHANNELS=NUM_CHANNELS, ONED=True)
        test_data = test_data[:, :, :]
        test_data_list.append(test_data)
        test_data_node = tf.constant(test_data_list[0])
        test_prediction = tf.nn.softmax(model(test_data_node, False))
        test_prs = test_prediction.eval()

        # Set the highest probability to 1.
        p_nums = numpy.argmax(test_prs, 1)

        # Using the principle of majority rule, select the most frequent drivers.
        freq_prs = []
        freq_who = []
        for i in range(len(p_nums)):
            freq_prs.append(p_nums[i])
            if len(freq_prs) > 0:
                count = Counter(freq_prs)
                freq_list = count.most_common(1)[0][0]
                freq_who.append(freq_list)
            else:
                i+=1

        # Outputs the result of detecting who is driver.
        acc = [0, 0, 0, 0]
        acc_graph_after=[]
        acc_graph=[]
        for f_num in freq_who:
            acc[f_num] += 1
            print 'test_prediction:', DRIVERS_NAME[f_num]

        for i in range(0, 4):
            acc[i] = float(float(acc[i]) / float(len(freq_who)))
        print acc

        print('unknown driver name: %s' % test_label)
        who_max = numpy.argmax(acc)
        test_who = DRIVERS_NAME[who_max]
        print 'who is driver?: ', test_who


        # Accuracy graph.
        # no use the principle of majority rule.
        n_af = 1
        for k in range(len(p_nums)):
            if p_nums[k] == LABEL_NUM:
                correct = float(n_af / (float(k + 1)))*100
                acc_graph.append(correct)
                n_af += 1
            elif p_nums[k]!=LABEL_NUM:
                miss = float((n_af-1) / (float(k + 1)))*100
                acc_graph.append(miss)


        # use the principle of majority rule.
        af = 1
        for k in range(len(freq_who)):
            if freq_who[k] == LABEL_NUM:
                correct = float(af / (float(k + 1)))*100
                acc_graph_after.append(correct)
                af += 1
            elif freq_who[k]!=LABEL_NUM:
                miss = float((af-1) / (float(k + 1)))*100
                acc_graph_after.append(miss)

    plt.figure()
    plt.plot(acc_graph,'bo-', label='no_after')
    plt.plot(acc_graph_after,'go-', label='Real time accuracy')
    plt.ylabel('accuracy (%)')
    plt.xlabel('Times (0.25s)')
    plt.legend()
    plt.title('Time check')
    plt.axis('on')
    plt.grid('on')
    plt.ylim(0,105)
    plt.show()


if __name__ == '__main__':
    tf.app.run()