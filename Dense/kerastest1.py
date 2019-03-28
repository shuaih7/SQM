from __future__ import absolute_import, division, print_function                    

import pathlib
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import seaborn as sns
   
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers

#from keras.optimizers import 

##print(tf.__version__)


###### Generates the Input Data ######
######################################

###### PART 1: Testing Data ######
XMIN = -1.0
XMAX = 1.0
XDEV = 50
YMIN = -1.0
YMAX = 1.0
YDEV = 50

Trainset = np.zeros(((XDEV+1)*(YDEV+1),2))
Tlabel = np.zeros(((XDEV+1)*(YDEV+1),1))
Testset = np.zeros((XDEV*YDEV,2))
Testlb = np.zeros((XDEV*YDEV,1))
for i in xrange(XDEV+1):
    for j in xrange(YDEV+1):
        Trainset[i*(XDEV+1)+j][0] = XMIN+i*(XMAX-XMIN)/XDEV 
        Trainset[i*(XDEV+1)+j][1] = YMIN+j*(YMAX-YMIN)/YDEV
        Tlabel[i*(XDEV+1)+j] = 1-Trainset[i*(XDEV+1)+j][0]**2-Trainset[i*(XDEV+1)+j][1]**2
        if i<XDEV and j<YDEV:
            Testset[i*XDEV+j][0] = Trainset[i*(XDEV+1)+j][0]+(XMAX-XMIN)/XDEV/2
            Testset[i*XDEV+j][1] = Trainset[i*(XDEV+1)+j][1]+(YMAX-YMIN)/YDEV/2
            Testlb[i*XDEV+j][0] = Tlabel[i*(XDEV+1)+j][0] ### This is not correct 

###### PART 2: Training Data ######
"""vmin = 0
vmax = 1
smin = 0.01
smax = 1
vdev = 100
sdev = 99
A1 = 3.3322
A2 = 12.829
a = 1.0
eta =1.0

dv = (vmax-vmin)/vdev
ds = (smax-smin)/sdev
Trainset = np.zeros(((vdev+1)*(sdev+1),2))
Tlabel = np.zeros(((vdev+1)*(sdev+1),1))
Testset = np.zeros((vdev*sdev,2))
Testlb = np.zeros((vdev*sdev,1))

for i in xrange(vdev+1):
    for j in xrange(sdev+1):
        Trainset[i*(sdev+1)+j][0] = vmin + i*dv
        Trainset[i*(sdev+1)+j][1] = smin + j*ds
        v0 = vmin + i*dv
        s0 = smin + j*ds
        Tlabel[i*(sdev+1)+j][0] = -1*eta/2*v0*(A1*np.power(2*a/s0,1.5)+A2*np.power(2*a/s0,0.5))
        if i<vdev and j<sdev:
            Testset[i*sdev+j][0] = Trainset[i*(sdev+1)+j][0] + dv/2
            Testset[i*sdev+j][1] = Trainset[i*(sdev+1)+j][1] + ds/2
            v0 = Testset[i*sdev+j][0]
            s0 = Testset[i*sdev+j][1]
            Testlb[i*sdev+j][0] = -1*eta/2*v0*(A1*np.power(2*a/s0,1.5)+A2*np.power(2*a/s0,0.5))
   """     

###### Plotting Test 1 ######
#############################
"""pX = np.linspace(-1,1,XDEV+1)
pY = np.linspace(-1,1,YDEV+1)
pZ = np.zeros(((XDEV+1),(YDEV+1)))

for i in xrange(XDEV+1):
    for j in xrange(YDEV+1):
        pZ[i][j] = Tlabel[i*(XDEV+1)+j]

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

Axes3D.plot_surface(pX,pY,pZ)
plt.show()"""


###### Training the Model ######
################################

rows = Trainset.shape[0]
cols = Trainset.shape[1]
penalty = 0.001

def build_model():
  model = keras.Sequential([
    layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(penalty)),
    layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(penalty)),
    #layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(1)  ## There is no activation function at the last layer!
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model


model = build_model()

EPOCHS = 8192

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

history = model.fit(
  Trainset, Tlabel, batch_size=1024, epochs=EPOCHS, validation_split = 0.0, verbose=0,
  callbacks=[PrintDot()])

###### Plot the Error Evolution ######
######################################

def plot_history(history):

### Plot the main absolute error ###
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  #plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
   #        label = 'Val Error')
  plt.ylim([-0.5,50])
  plt.legend()
  
### Plot the main squared error ###
  """plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([-1,100])
  plt.legend()"""
  plt.show()


plot_history(history)

#test_predictions = model.predict(Testset).flatten()
test_predictions = model.predict(Testset)
#print (Testlb - test_predictions)

###### Write Data to File ######
################################

f = open("Prabolic.txt","w")
for i in xrange(XDEV):
    for j in xrange(YDEV):
        f.write("%6.4f " % (test_predictions[i*YDEV+j][0]))
        #f.write("%6.4f " % (Testset[i*sdev+j][1]))
    f.write("\n")
f.close()

