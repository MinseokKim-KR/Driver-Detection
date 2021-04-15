import numpy as np
import pandas as pd
import os
import tensorflow as tf

# user configuration.
DATE = '170217'
DRIVERS = 'CS,HH,JH,TW'
TEST_NUM = 1
SAVE_DIR = 'extract_train_data'


# user inputs.
tf.app.flags.DEFINE_string('drivers', DRIVERS, "Set")
tf.app.flags.DEFINE_string('date', DATE, "Set")
tf.app.flags.DEFINE_integer('test_num', TEST_NUM, "Set")
tf.app.flags.DEFINE_string('save_dir', SAVE_DIR, "Set")
FLAGS = tf.app.flags.FLAGS

DRIVERS = FLAGS.drivers.split(',')
TEST_NUM = FLAGS.test_num
BASE = FLAGS.date
SAVE_DIR = FLAGS.save_dir



# divided data file merge.
for k in range(len(DRIVERS)):
    dirName = './raw_data/' + DATE + '/' + DRIVERS[k]
    filenames = []
    if os.path.exists(dirName):
        for fn in os.listdir(dirName):
            filenames.append(dirName + '/' + fn)
            filenames.sort()
        df=filenames
        df_concat=[]
        print "File lenth", len(df)


    for i in range(len(df)):
        x = pd.read_csv(df[i])
        if len(df_concat) != 0:
            df_concat = [df_concat,x]
            df_concat = pd.concat(df_concat)
        else:
            df_concat = x


    # extract sensor data.
    df_concat = df_concat[['EMS2::PV_AV_CAN', "ESP2::CYL_PRES", "EMS1::VS"]]
    df_concat.to_csv('./%s/%s/%s_test%s_%s' %(SAVE_DIR, DRIVERS[k], DATE, TEST_NUM, DRIVERS[k]), index=False)
    df_concat = pd.read_csv('./%s/%s/%s_test%s_%s' %(SAVE_DIR, DRIVERS[k], DATE, TEST_NUM, DRIVERS[k]))

    # 0.25s divide.
    df_concat.index = pd.to_datetime(df_concat.index, unit='s')
    df_time_merge = df_concat.resample('520S',how='mean')


    # remove nan.
    save1 = np.array([])
    save2 = np.array([])
    save3 = np.array([])
    for j in df_time_merge['EMS2::PV_AV_CAN']:
        if ~np.isnan(j):
            save1 = np.append(save1, j)

    for j in df_time_merge["ESP2::CYL_PRES"]:
        if ~np.isnan(j):
            save2 = np.append(save2, j)

    for j in df_time_merge["EMS1::VS"]:
        if ~np.isnan(j):
            save3 = np.append(save3, j)

    save_all = np.array([])
    list1 = [save1,save2,save3]
    for i in range(len(save3)):
        for j in range(0,3):
            save_all = np.append(save_all,list1[j][i])
    save_all = np.reshape(save_all,(-1,3))


    # save preprocessed data.
    np.savetxt('./%s/%s/%s_test%s_%s' %(SAVE_DIR, DRIVERS[k], DATE, TEST_NUM, DRIVERS[k]), save_all, delimiter=',')
    print "Save preprocessed file: driving date= %s, driver= %s" % (DATE, DRIVERS[k])
    print "Save directory: %s" % (SAVE_DIR)



