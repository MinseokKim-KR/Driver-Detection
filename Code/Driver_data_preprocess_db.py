import MySQLdb
import MySQLdb.cursors
import numpy as np
import datetime
import tensorflow as tf
import time
server_db_name = 'test_table18'

db = MySQLdb.connect(host = "203.246.114.176",
		     user = "hyundai",
		     passwd = "autron",
		     port = 3306,
		     charset = "utf8",
		     cursorclass = MySQLdb.cursors.DictCursor,
		     db = "test")
# IP : 203.246.114.176, databasename : testt

cur = db.cursor(MySQLdb.cursors.DictCursor) # setup the cursor to use database
X_width = 2 # make data to 300 * 2

def make_x(table): #make input data to nomalization and reshape 300 * 2
    MAX = [0, 0]
    MIN = [0, 0] #set up the max and min of the two sensor
    x_data=[]
    for i in range(0,2):
        for j in range(len(table)):
            MAX[i] = max(MAX[i], table[j][i])
            MIN[i] = min(MIN[i], table[j][i]) #insert max and min
    temp = [0, 0]
    # temp = np.array(temp) #make temp and it will be train or test data
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

def input_data(id, time1, cur): #read from Database
    # time1 = str(time1) #time = 75 sec
    atime=5
    row2 = []
    # command = "select VehicleID, Time, uuid, Data" \
    #           " from test_table13\
    #             where VehicleID = '" + id + "' and Entry_time >= DATE_ADD(NOW(),INTERVAL - " + time + " SECOND)\
    #                   and Entry_time <= DATE_ADD(NOW(), INTERVAL 0 SECOND)\
    #                   order by Entry_time; "
    # command = "select VehicleID, Time, uuid, Data from test_table12 " \
    #           "where VehicleID = '100' and Entry_time >= DATE_ADD('2017-09-07 19:21:33',INTERVAL -100 SECOND) " \
    #           "and Entry_time <= DATE_ADD('2017-09-07 17:26:46', INTERVAL 0 SECOND) order by Entry_time;"
    i = 1
    while len(row2)<900 :
        # command = "select VehicleID, Time, uuid, Data" \
        #           " from test_table18\
        #             where VehicleID = '" + id + "' and Entry_time >= DATE_ADD(NOW(),INTERVAL - " + str(time1) + " SECOND)\
        #                   and Entry_time <= DATE_ADD(NOW(), INTERVAL 0 SECOND) order by Number; "
        # command = "select VehicleID, Time, uuid, Data from test_table18 " \
        #           "where VehicleID = '" + id + "' and Number >= " +str(int(int(time1)*int((number))))+" and Number <= 900+" + str(int(int(time1)*int((number))))+";"
        command = "select VehicleID, Time, uuid, Data" \
                  " from " + server_db_name + " where VehicleID = '" + id + "' and Entry_time >= DATE_ADD(NOW(),INTERVAL - " + str(time1) + " SECOND)\
                    order by Number limit 900"
        cur.execute(command)
        # row1 = cur.fetchone()

        # print(row1["numb"])
        # db.commit()
        # time.sleep(1)
        #
        cur.execute(command)
        row2 = cur.fetchall()
        # print command
        # print row2
        # print 'len(row)', len(row2)
        # print row2[0]
        time1=time1+atime
        db.commit()
        time.sleep(0.5)

    return row2



def make_new_matrix(row):
    matrix = np.zeros((300,2))
    # if len(row)<900 :
    #     for i in range(len(row)/3):
    #         for j in range(0, 2):
    #             matrix[i][j] = row[3 * i + j]['Data']
    #     for i in range(len(row)/3, 300):
    #         matrix[i][0] = row[len(row) - 3]['Data']
    #         matrix[i][1] = row[len(row) - 2]['Data']
    # else:
    #     for i in range(0,300):
    #         for j in range(0,2):
    #             matrix[i][j]=row[3*i+j]['Data']
    # for i in range(0,300):
    #     for j in range(0,2):
    #         matrix[i][j]=row[3*i+j]['Data']
    for i in range(0, 300):
        for j in range(0,3):
            if row[3 * i + j]['uuid'] == '0':
                matrix[i][0] = row[3 * i + j]['Data']
            elif row[3 * i + j]['uuid'] == '1':
                matrix[i][1] = row[3 * i + j]['Data']

    return matrix

def server_Data(Vehicle = None, Time = None, number = None):
    if Vehicle==None:
        Vehicle = '100'
    if Time == None:
        Time = 75
    row = input_data(Vehicle, Time, cur) #all row
    row = list(row)
    temp = []
    test_data=[]
    temp = make_new_matrix(row)
    test_data = make_x(temp)
    test_data = test_data.astype(np.float32)

    return test_data
