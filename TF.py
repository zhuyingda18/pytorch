# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
import datetime
from tensorflow.python.training.moving_averages import assign_moving_average


mnist = fetch_mldata('MNIST original', data_home='./data/')
mnist_x, mnist_y = mnist.data, mnist.target
mnist_x = mnist_x / 255.

train_x, test_x, train_y, test_y = train_test_split(mnist_x, mnist_y, test_size=0.2, random_state=2)
train_y = np.eye(10)[train_y.astype("int")]
test_y = np.eye(10)[test_y.astype("int")]

train_n = train_x.shape[0]
test_n = test_x.shape[0]

K=1000
L=1000
M=500
N=500
epsilon = 1e-4

W1 = tf.Variable(tf.truncated_normal([28 * 28, K], stddev=0.1))
B1 = tf.Variable(tf.zeros([K]))
W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
B2 = tf.Variable(tf.zeros([L]))
W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B3 = tf.Variable(tf.zeros([M]))
W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B4 = tf.Variable(tf.zeros([N]))
keep_prob = tf.placeholder("float")      #dropout
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

w2_BN = tf.Variable(np.random.normal(size=(K,K)).astype(np.float32))
scale2 = tf.Variable(tf.ones([K]))
beta2 = tf.Variable(tf.zeros([K]))     #Bachnorm


X = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])
X = tf.reshape(X, [-1, 28 * 28])

#Layer1
Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)  #Linear
# z2_BN = tf.matmul(Y1,w2_BN)             #Bachnorm
# batch_mean2, batch_var2 = tf.nn.moments(z2_BN,[0])
# BN2 = tf.nn.batch_normalization(z2_BN,batch_mean2,batch_var2,beta2,scale2,epsilon)
# Y1_BN = tf.nn.relu(BN2)
Y1_BN = tf.nn.dropout(Y1, keep_prob)       #Dropout

#Layer2
Y2 = tf.nn.relu(tf.matmul(Y1_BN, W2) + B2) #Linear
drop2 = tf.nn.dropout(Y2, keep_prob)       #Dropout

#Layer3
Y3 = tf.nn.relu(tf.matmul(drop2, W3) + B3) #Linear
drop3 = tf.nn.dropout(Y3, keep_prob)       #Dropout

#Layer3
Y4 = tf.nn.relu(tf.matmul(drop3, W4) + B4) #Linear
drop4 = tf.nn.dropout(Y4, keep_prob)       #Dropout

#Output
pred = tf.nn.softmax(tf.matmul(drop4, W5) + B5)  #Linear


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
accu = tf.reduce_mean(tf.cast(corr, tf.float32))
init = tf.global_variables_initializer()

sess=tf.InteractiveSession()
sess.run(init)

n_epoch = 200
batchsize = 100

for n in range(n_epoch):
    begin = datetime.datetime.now()
    perm = np.random.permutation(train_n)
    for i in range(0, train_n, batchsize):
        x = train_x[perm[i: i + batchsize]]
        t = train_y[perm[i: i + batchsize]]
        batch_xs, batch_ys = x,t
        sess.run(train_step, feed_dict={X: batch_xs, y_: batch_ys,keep_prob:0.7})
    end = datetime.datetime.now()
    print("epoch = "+str(n+1),end='')
    print("  Train Accuracy：%f" %accu.eval(feed_dict={X:train_x, y_:train_y,keep_prob:1}, session=sess),end='')
    print("  Test Accuracy：%f" %accu.eval(feed_dict={X:test_x,y_:test_y,keep_prob:1},session=sess),end='')
    print("  Time: " +str(end-begin))



