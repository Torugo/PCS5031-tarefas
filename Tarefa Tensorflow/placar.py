
import numpy as np
import tensorflow as tf

import glob

a = tf.placeholder(tf.int32, shape=(90, 2), name='a')
teamA = tf.Variable(0, name="TeamA")
teamB = tf.Variable(0, name="TeamB")
win = tf.constant(3, dtype=tf.int32)
drew = tf.constant(1, dtype=tf.int32)

#operations
teamAWin = tf.assign_add(teamA,win)
teamADrew = tf.assign_add(teamA,drew)
teamBWin = tf.assign_add(teamB,win)
teamBDrew = tf.assign_add(teamB,drew)

placar = tf.reduce_sum(a, axis=0, name='placar')

with tf.Session() as sess:
    sess.run(teamA.initializer)
    sess.run(teamB.initializer)
    files = glob.glob("./data/*.txt")
    for game in files:
        data = np.loadtxt(game)
        a_val = a.eval(feed_dict={a: data})
        placar_val = placar.eval(feed_dict={a: data})
        print (placar_val)
        if placar_val[0] > placar_val[1]:
            sess.run(teamAWin)
            print("A win")
        elif placar_val[0] < placar_val[1]:
            sess.run(teamBWin)
            print("B Win")
        else:
            sess.run(teamADrew)
            sess.run(teamBDrew)
            print("Drew")

    aPoints = teamA.eval()
    bPoints = teamB.eval()
    print("team A pontuation =", teamA.eval())
    print("team B pontuation =", teamB.eval())
    print()
    if aPoints > bPoints:
        print("Team A is the Champion")
    elif bPoints < aPoints:
        print("Team B is the Champion")
    else:
        print("The teams has the same pontuation")