# This is just a test for the linear regression case.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Part1: DEfining some basic parameters:
learning_rate = 0.01
epoch = 1000
display = 50
n_data = 20


# Part2: Basic operations
x_input = np.linspace(-1.,1.,n_data)
n_samples = x_input.shape[0]
y_input = 2*x_input + np.random.normal(0., 0.2, n_samples)

# Make sure that all of the values needed to be modified by TF show be initialized as follows
#X = tf.placeholder(tf.float32)
#Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal(shape=[1],mean=0.0,stddev=0.2,dtype=tf.float32))
b = tf.Variable(tf.random_normal(shape=[1],mean=0.0,stddev=0.2,dtype=tf.float32))

y_pred = tf.add(tf.multiply(W,x_input),b)

#plt.figure(1)
#plt.plot(x_input,y_input,'bo',label='True Data')
#plt.axis('equal')
#plt.legend(loc='lower right')
#plt.show()

lost = tf.reduce_mean(tf.square(y_input-y_pred))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(lost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in xrange(epoch):
    #lst = sess.run(lost, feed_dict={X:x_input, Y:y_input})
    lst = sess.run(lost)
    #opt = sess.run(optimizer, feed_dict={X:x_input, Y:y_input})
    opt = sess.run(optimizer)
    if step%display == 0:
        print lst

W_disp = sess.run(W)
b_disp = sess.run(b)
plt.figure(2)
plt.plot(x_input,y_input,'bo',label='True Data')
plt.plot(x_input,W_disp*x_input+b_disp,'ro',label='Training Data')
plt.axis('equal')
plt.legend(loc='lower right')
plt.show()

sess.close()

