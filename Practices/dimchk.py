import numpy as np
import tensorflow as tf

# Check the dimension definitions of zeros([ ])

ZERO = np.zeros([3,2])
index1 = ZERO.shape[0]
index2 = ZERO.shape[1]

SUB = [1., 1.]
for i in range(index1):
    for j in range(index2):
        ZERO[i,j] = i*index2+j

ZERO[0,:] = SUB        
print ZERO
