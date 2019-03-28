# This practice will try for a matrix multiplication case

### Import necessary packages
import numpy as np
import tensorflow as tf

tf.reset_default_graph()

###Input data
print "Enther an index: "
index = input()

  # Use the normal python code, do not pass to tensorflow
n = 6
row = 2
col = n/row
x = np.zeros([n])
for i in xrange(n):
    x[i] = i

if  index > 0:
    x = x.reshape((row,col))
    b = np.array([1,1,2])    
    c = np.matmul(x,b)
    print c
else:
    x = tf.reshape(x,(row,col))
    b = tf.reshape(np.array([1.,1.,2.]),(col,1))

    c = tf.matmul(x,b)
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    writer = tf.summary.FileWriter('/home/shuai/',sess.graph)
    print sess.run(c)

