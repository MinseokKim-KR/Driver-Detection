# edited by SJLIM

from collections import Counter
import numpy
import numpy as np
import tensorflow as tf
import Driver_data_preprocess_db as extractData ###
import timeit
import matplotlib.pyplot as plt
import time
import MySQLdb.cursors

db = MySQLdb.connect(host = "203.246.114.176",
		     user = "hyundai",
		     passwd = "autron",
		     port = 3306,
		     charset = "utf8",
		     cursorclass = MySQLdb.cursors.DictCursor,
		     db = "test")

cur = db.cursor(MySQLdb.cursors.DictCursor)

# data set paramPatheter
PATH = 'train_dir'
DATA_TYPES = ''
LABEL = '0,1,2,3'
DRIVERS_NAME = ['CS', 'HH', 'JH', 'TW']
LABEL_NUM = 0

test_label = 0

Vehicle = '100'
Time =  75 #read time of the DB

# model parameter
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

# user inputs
# Path = './test_input/'
tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
tf.app.flags.DEFINE_string('data_types', DATA_TYPES, "Set the data type")
tf.app.flags.DEFINE_string('labels', LABEL, "Set the labels")


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
tf.app.flags.DEFINE_string('Vehicle', Vehicle, "Set the Vehicle")
FLAGS = tf.app.flags.FLAGS


# model parameter
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

Vehicle = FLAGS.Vehicle
#check the Vehicle number
# print "Vehicle ", Vehicle

#initialize the fullyconnected layer
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

#initialize the convolution layer
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


def main(argv=None):  # pylint: disable=unused-argument
    if FLAGS.self_test:
        # print 'Running self-test.'
        num_epochs = 1
    else:
        # Extract it into numpy arrays.

        LABELS = FLAGS.labels.split(',')
        NUM_LABELS = len(set(LABELS))

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.get_variable("C1", shape=[CONV_1_H, NUM_CHANNELS, CONV_1_D],
                                    initializer=xavier_init2(CONV_1_H, NUM_CHANNELS, CONV_1_D))
    conv1_biases = tf.Variable(tf.zeros([CONV_1_D]))

    conv2_weights = tf.get_variable("C2", shape=[CONV_2_H, CONV_1_D, CONV_2_D],
                                    initializer=xavier_init2(CONV_2_H, CONV_1_D, CONV_2_D))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[CONV_2_D]))

    conv3_weights = tf.get_variable("C3", shape=[CONV_3_H, CONV_2_D, CONV_3_D],
                                    initializer=xavier_init2(CONV_3_H, CONV_2_D, CONV_3_D))
    conv3_biases = tf.Variable(tf.constant(0.1, shape=[CONV_3_D]))

    CONV1H = CONV_1_H
    CONV2H = CONV_2_H
    if CONV_1_PADDING == 'SAME':
        CONV1H = 1
    if CONV_2_PADDING == 'SAME':
        CONV2H = 1
    WH = ((NUM_ROWS - CONV1H + 1) / (POOL_1) -CONV2H+ 1) / POOL_2 * CONV_2_D + 64
    # print WH

    fc1_weights = tf.get_variable("W1", shape=[WH, FULL_D], initializer=xavier_init(WH, FULL_D))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[FULL_D]))

    fc2_weights = tf.get_variable("W2", shape=[FULL_D, NUM_LABELS], initializer=xavier_init(FULL_D, NUM_LABELS))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.

    #model is the important part. The test_data go through model and you can prediction it.
    def model(data, train):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].

        # start first convolution layer
        conv = tf.nn.conv1d(data, conv1_weights, stride=1, padding="SAME")

        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.

        if True:
            pool = tf.layers.max_pooling1d(relu, POOL_1, strides=POOL_1, padding='SAME')

        else:
            pool = relu

        #start second convolution layer

        conv = tf.nn.conv1d(pool, conv2_weights, stride=1, padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))

        if True:
            pool = tf.layers.max_pooling1d(relu, POOL_2, strides=POOL_2, padding='SAME')
        else:
            pool = relu

        # start third convolution layer
        conv = tf.nn.conv1d(pool, conv3_weights, stride=1, padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))

        if True:
            pool = tf.layers.max_pooling1d(relu, POOL_3, strides=POOL_3, padding='SAME')
        else:
            pool = relu
            # print pool

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        # print pool_shape
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2]])

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        # print 'before test work'
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
            # print 'train work'
        return tf.matmul(hidden, fc2_weights) + fc2_biases


    freq_prs = [] #before treatment
    freq_who = [] #after treatment

    saver = tf.train.Saver(tf.all_variables()) #read all parameter of the weights and biases
    ccount=0
    while 1:
        with tf.Session() as s:
                # Run all the initializers to prepare the trainable parameters.
                tf.initialize_all_variables().run()
                saver.restore(s, PATH+'/'+'driver.ckpt-2499')

                test_data_list = []
                test_data = extractData.server_Data(Vehicle=Vehicle, Time=Time) #read test_data of the server
                ccount +=1 #count the number of the test number
                test_data_list.append(test_data)
                test_data_node = tf.constant(test_data_list[0])
                test_data2 = [test_data_node] #make matrix to fourth dimension
                test_prediction = tf.nn.softmax(model(test_data2, False))#insert the test_data to the model
                test_prs = test_prediction.eval() #calculate the probability

                p_nums = numpy.argmax(test_prs, 1) #make the maximum of the probability to 1 and others 0
                # print DRIVERS_NAME[p_nums]
                for i in range(len(p_nums)):
                    freq_prs.append(p_nums[i])
                    if len(freq_prs) > 0:
                        count = Counter(freq_prs)
                        freq_list = count.most_common(1)[0][0]
                        freq_who.append(freq_list)
                    else:
                        i+=1

                acc = [0, 0, 0, 0]
                # for f_num in freq_who:
                #     acc[f_num] += 1
                #     print 'test_prediction:', DRIVERS_NAME[f_num]

                #check before the treatment person
                print 'test_prediction:', DRIVERS_NAME[freq_who(len(freq_who)-1)]

                for i in range(0, 4):
                    acc[i] = float(float(acc[i]) / float(len(freq_who)))

                who_max = numpy.argmax(acc)
                test_who = DRIVERS_NAME[who_max]
                result = DRIVERS_NAME[test_label]
                # print freq_who
                # print acc

                #print result to the screen
                print 'who is driver?: ', test_who # after treatment person
                print 'number', ccount
                # print 'result?: ', result

                #write to the result table
                command = 'INSERT INTO result (VehicleID,  PersonID) VALUES (' + str(Vehicle) + ',' + str(who_max) + ')'
                cur.execute(command)
                db.commit()
                time.sleep(0.5)

if __name__ == '__main__':
    tf.app.run()