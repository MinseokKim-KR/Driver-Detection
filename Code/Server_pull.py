import MySQLdb.cursors
import time
import os
import pandas as pd
import tensorflow as tf

server_db_name = 'test_table17'
client_db_name= 'test_table18'

tf.app.flags.DEFINE_string('server_db_name', server_db_name, "Set the server_db_name")
tf.app.flags.DEFINE_string('client_db_name', client_db_name, "Set the client_db_name")

FLAGS = tf.app.flags.FLAGS

server_db_name = FLAGS.server_db_name
client_db_name = FLAGS.client_db_name

# data set parameter

db = MySQLdb.connect(host = "203.246.114.176",
		     user = "hyundai",
		     passwd = "autron",
		     port = 3306,
		     charset = "utf8",
		     cursorclass = MySQLdb.cursors.DictCursor,
		     db = "test")
#db = test, t
cur = db.cursor(MySQLdb.cursors.DictCursor)

while(1) :
    command="select Number from "+client_db_name+" order by Number desc limit 1;"
    cur.execute(command)
    row = cur.fetchall()
    if len(row)==0:
        c_number=0
    else:
        c_number = int(row[0]['Number'])


    ###### read end of the record

    # command="select Number from "+server_db_name+" order by Number desc limit 1;"
    # cur.execute(command)
    # row2 = cur.fetchall()
    # if len(row2)==0:
    #     s_number=1
    # else:
    #     s_number = int(row2[0]['Number'])

    print "c_number",c_number
    # print "s_number", s_number

    # command = "select VehicleID, Time, uuid, Data from "+server_db_name +\
    #           " where Number >= "+str(c_number)+ " and Number <= "+ str(s_number)+" order by Number;"
    command = "select VehicleID, Time, uuid, Data from " + server_db_name + \
              " where Number > " + str(c_number) + " order by Number;"
    cur.execute(command)
    new_row = cur.fetchall()
    print new_row

    for i in range(len(new_row)):
        q = "insert into "+client_db_name+" (VehicleID, Time, uuid, Data) values (%s, %s, %s, %s)"
        cur.execute(q, (new_row[i]['VehicleID'], new_row[i]['Time'], new_row[i]['uuid'], new_row[i]['Data']))
    db.commit()
    time.sleep(0.5)