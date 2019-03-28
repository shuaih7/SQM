# This is a practice for reading data from files

import tensorflow as tf
import numpy as np

f = open("sample.txt","r")
l = [[float(num) for num in line.split(' ')] for line in f]
print (l)
f.close()
