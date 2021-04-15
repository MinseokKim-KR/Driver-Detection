import csv
import os
import numpy
import numpy as np
import random
from joblib import Parallel, delayed
import multiprocessing


# user configuration
BASE = './extract_train_data'
MIN_ROWS = 32

Drivers = ['CS','HH','JH','TW']
X_width = 2
Labels = {}


def get_files(drivers=None):
    if drivers is None or len(drivers) == 0:
        drivers = Drivers

    filenames = []
    filelabels = []

    dirName = BASE
    for driver in drivers:
        dirName1 = os.path.join(dirName, driver)
        print("driver = %s" % (driver))

        if os.path.exists(dirName1):
            for fn in os.listdir(dirName1):
                filenames.append(dirName1 + '/' + fn)
                filelabels.append(Labels[driver])

    print "found %d files" % len(filenames)
    return filenames, filelabels


def fileReader(fn):
    reader = csv.reader(open(fn, "rb"), delimiter=',')
    x = list(reader)
    result = numpy.array(x).astype('float')
    print '%s has %d rows.' % (fn, result.shape[0])
    return result


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

# groups = None, personIDs = None,
def extract_data_oned(drivers=None, labels=None, numRows=None, data_types = None, numData=None, mode=None, NUM_CHANNELS=None,
                     ONED=True, DATA_SIZE=None):

    NUM_LABELS = len(set(labels))
    print NUM_LABELS
    for s, l in zip(drivers, labels):
        Labels[s] = l

    filenames, filelabels = get_files(drivers)

    results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(fileReader)(fn) for fn in filenames)

    if ONED:
        data = numpy.ndarray(shape=(numData, numRows, NUM_CHANNELS), dtype=numpy.float32)
    else:
        data = numpy.ndarray(shape=(numData, numRows, DATA_SIZE, 1), dtype=numpy.float32)
    dataLabel = numpy.zeros(shape=(numData, NUM_LABELS), dtype=numpy.float32)

    numDataCount = 0
    while numDataCount < numData:
        res_idx = random.randint(0, len(results) - 1)
        max_row_idx = results[res_idx].shape[0] - numRows

        if max_row_idx < MIN_ROWS:  # this value should be at least 0
            continue

        if mode is 'train':
            row_idx = random.randint(0, max_row_idx * 4 / 5)  ## train : 4 / 5
        elif mode is 'validate':
            row_idx = random.randint(max_row_idx * 4 / 5, max_row_idx)  ## validate : 1 / 5
        if ONED:
            data[numDataCount, :, :] = results[res_idx][row_idx:row_idx + numRows, 0:NUM_CHANNELS]
            data[numDataCount, :, :] = make_x(data[numDataCount, :, :])
        else:
            data[numDataCount, :, :, 0] = results[res_idx][row_idx:row_idx + numRows, 0:DATA_SIZE]
        filelabels[res_idx] = int(filelabels[res_idx])


        dataLabel[numDataCount, filelabels[res_idx]] = 1.0
        numDataCount = numDataCount + 1
        print numDataCount

    return data, dataLabel


    if __name__ == '__main__':
        extract_data_oned()



