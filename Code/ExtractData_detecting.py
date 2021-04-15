import csv
import os
import numpy
import numpy as np
import random
from joblib import Parallel, delayed
import multiprocessing


# user configuration
BASE = './extract_test_data'

MIN_ROWS = 32
NUM_CHANNELS= 2
Drivers = ['CS','HH','JH','TW']
Labels = {}
X_width=2

# Batch Normalization.
def make_x(table): # make input data to nomalization and reshape 300 * 2
    MAX = [0, 0]
    MIN = [0, 0] #set up the max and min of the two sensor
    x_data=[]
    for i in range(0,2):
        for j in range(len(table)):
            MAX[i] = max(MAX[i], table[j][i])
            MIN[i] = min(MIN[i], table[j][i]) #insert max and min
    temp = [0, 0]
    # make temp and it will be train or test data
    for i in range(len(table)):
        for j in range(0, 2):
            if MAX[j]-MIN[j] != 0 :
                temp[j] = float(table[i][j]- MIN[j])/float(MAX[j]-MIN[j])
            else :
                temp[j] = 0
        x_data = np.append(x_data, temp)
        temp = [0, 0]

    x_data = np.reshape(x_data, (-1, X_width))

    return x_data


# extractData
def get_files(drivers=None, data_types=None):
    if drivers is None or len(drivers) == 0:
        drivers = Drivers

    filenames = []

    dirName = BASE

    for driver in drivers:
        dirName1 = os.path.join(dirName, driver)
        print("driver = %s" % (driver))
        if os.path.exists(dirName1):
            for fn in os.listdir(dirName1):
                filenames.append(dirName1 + '/' + fn)

    print "found %d files" % len(filenames)
    print "filenames", filenames

    return filenames

def fileReader(fn):
    reader = csv.reader(open(fn, "rb"), delimiter=',')
    x = list(reader)
    result = numpy.array(x).astype('float')
    print '%s has %d rows.' % (fn, result.shape[0])
    return result



def extract_data_oned(drivers=None, labels=None, numRows=None, data_types = None, numData=None, mode=None, NUM_CHANNELS=None,
                     ONED=True, DATA_SIZE=None, Shifting_strides=None):

    filenames = get_files(drivers)

    results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(fileReader)(fn) for fn in filenames)

    if numData is None:
        numData = len(results[0])

    if numRows is None:
        numRows = len(results[0])

    if ONED:
        data = numpy.ndarray(shape=(numData, numRows, NUM_CHANNELS), dtype=numpy.float32)

    numDataCount = 0
    row_idx=0
    res_idx=0
    while numDataCount < numData:

        max_row_idx = results[res_idx].shape[0] - numRows

        if max_row_idx - row_idx < MIN_ROWS:  # this value should be at least 0
            break

        if ONED:
            data[numDataCount, :, :] = results[res_idx][row_idx:row_idx + numRows, 0:NUM_CHANNELS]
            data[numDataCount, :, :] = make_x(data[numDataCount, :, :])
        else:
            data[numDataCount, :, :, 0] = results[res_idx][row_idx:row_idx + numRows, 0:DATA_SIZE]

        numDataCount = numDataCount + 1

        # Time shifting is 'Shifting_strides'.
        row_idx = row_idx + Shifting_strides
    return data


    if __name__ == '__main__':
        extract_data_oned()


