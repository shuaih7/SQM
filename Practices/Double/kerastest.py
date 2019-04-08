from __future__ import absolute_import, division, print_function                    

import pathlib
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import seaborn as sns
   
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

#print(tf.__version__)


###### Generates the Input Data ######
######################################

###### PART 1: Importing Data ######
Train = np.loadtxt('train.txt')
Test = np.loadtxt('test.txt')
Label = np.loadtxt('trainlabel.txt')
TLabel = np.loadtxt('testlabel.txt')


###### Settings before the Training ######
##########################################

RSP0 = Label.shape[0]
#RSP1 = TLabel.shape[0]
#Label.reshape((RSP0,1))
#TLabel.reshape((RSP1,1))
"""lb0 = np.zeros((RSP0,1))
lb1 = np.zeros((RSP0,1))
for i in xrange(RSP0):
    lb0 = Label[i,0]
    lb1 = Label[i,1]"""

learning_rate = 0.01
penalty = 0.001
EPOCHS = 4096
BATCH_SIZE = 512
actfunc = tf.nn.relu
regularizer = None
optimizer = tf.keras.optimizers.RMSprop(learning_rate)

###### Training the Model ######
################################
InPars = Train.shape[1]
inputs = Input(shape=(InPars,))

hidden = Dense(128,activation=actfunc,kernel_regularizer=regularizer)(inputs)
hidden = Dense(64,activation=actfunc,kernel_regularizer=regularizer)(hidden)
hidden = Dense(32,activation=actfunc,kernel_regularizer=regularizer)(hidden)
prediction = Dense(2)(hidden)

model = Model(inputs=inputs, outputs=prediction)
model.summary()
model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['mean_squared_error','mean_absolute_error'])

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

history = model.fit(Train,Label,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_split=0.0,verbose=0,callbacks=[PrintDot()])


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

train_predictions = model.predict(Train)
#train_predictions.reshape((RSP0,1))

test_predictions = model.predict(Test)
#test_predictions.reshape((RSP1,1))

###### Calculate the Mean Relative Error (MRE) ######
#####################################################
"""diff = 0.0
for i in xrange(RSP1):
    diff += np.absolute(test_predictions[i]-TLabel[i])

difft = 0.0
for j in xrange(RSP0):
    difft += np.absolute(train_predictions[j]-Label[j])

error = diff/np.sum(np.absolute(TLabel))
np.savetxt('output.txt',test_predictions,fmt='%8.6f')

errort = difft/np.sum(np.absolute(Label))

print("\nThe mean relative error of the training set is",errort)
print("\nThe mean relative error of the test set is",error)
print("\nRSP0 =",RSP0,"RSP1 =",RSP1)"""
np.savetxt('output.txt',test_predictions,fmt='%8.6f')
