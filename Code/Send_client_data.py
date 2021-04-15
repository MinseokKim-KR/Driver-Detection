import pickle
import socket
import sys
import MySQLdb
import MySQLdb.cursors
import time
import os
import pandas as pd
import tensorflow as tf

name = 'CS'
Vehicle=100
time1=0
sensorID=0
Data=0
person = 77
dtime = 0.5
Drivers = ['CS','HH','JH','TW']
BASE = './extract_test_data'
Names = ["EMS2::PV_AV_CAN", "ESP2::CYL_PRES", "EMS1::VS"]

tf.app.flags.DEFINE_string('names', name, "Set the name")
tf.app.flags.DEFINE_integer('Vehicle', Vehicle, "Set Vehicle")
FLAGS = tf.app.flags.FLAGS

name = FLAGS.names
Vehicle = FLAGS.Vehicle



db = MySQLdb.connect(host = "203.246.114.176",
		     user = "root",
		     passwd = "room#101",
		     port = 3306,
		     charset = "utf8",
		     cursorclass = MySQLdb.cursors.DictCursor,
		     db = "test")

cur = db.cursor(MySQLdb.cursors.DictCursor)


def get_files(Drivers=None):

    filenames = []
    filelabels = []
    dirName = BASE


    dirName1 = os.path.join(dirName, name)

    if os.path.exists(dirName1):
        for fn in os.listdir(dirName1):
            filenames.append(dirName1 +'/' + fn)
            filenames.sort()
    print "found %d files" % len(filenames)
    print filenames
    return filenames

filenames = get_files()
j=0
name=0
for z in range(0,len(filenames)):

    # csv =  pd.read_table(filenames[z],sep=',',names=["EMS2::PV_AV_CAN", "ESP2::CYL_PRES", "EMS1::VS", '4', '5', '6', '7', '8', '9', '10'])
    csv = pd.read_table(filenames[z], sep=',',names=["EMS2::PV_AV_CAN", "ESP2::CYL_PRES", "EMS1::VS"])

    # while 1:
    #     for i in range(0, len(csv)):
    #         for j in range(0,3):
    #             data = csv[names[j]][i]
    #             command = 'INSERT INTO test_table6 (VehicleID, PersonID, Time, uuid, Data) VALUES (' + str(port) +  ',' +   str(name) + ',' + str(
    #                             time1) + ',' + str(j) + ',' + str(data) + ')'
    #             print(command)
    #             cur.execute(command)
    #             db.commit()
    #             time1 = time1 + 1
    #             # sensorID = sensorID + 1
    #             # Data = Data + 1
    #             # time.sleep(0.5)
    # for i in range(0, len(csv)):
    #     for j in range(0, 3):
    #         data = csv[names[j]][i]
    #         command = 'INSERT INTO test_table0 (VehicleID, PersonID, Time, uuid, Data) VALUES (' + str(
    #             port) + ',' + str(name) + ',' + str(
    #             time1) + ',' + str(j) + ',' + str(data) + ')'
    #         print(command)
    #         cur.execute(command)
    #         db.commit()
    #         time1 = time1 + 1
    #         # sensorID = sensorID + 1
    #         # Data = Data + 1
    #         time.sleep(0.5)

    for i in range(0, len(csv)):
        for j in range(0, 3):
            data = csv[Names[j]][i]
            command = 'INSERT INTO test_table17 (VehicleID,  Time, uuid, Data) VALUES (' + \
                      str(Vehicle) + ',' + str(time1) + ',' + str(j) + ',' + str(data) + ')'
            print(command)
            cur.execute(command)
            db.commit()
            time1 = time1 + 1
            # sensorID = sensorID + 1
            # Data = Data + 1
            time.sleep(0.1)



