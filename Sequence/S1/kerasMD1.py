from __future__ import absolute_import, division, print_function                    

import pathlib
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
   
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model, load_model

import time  # import the timing function

##print(tf.__version__)


###### Generates the Input Data ######
######################################

###### PART 1: Testing Data ######

###### PART 2: Training Data ######
Trainset = np.loadtxt('train.txt')
Testset = np.loadtxt('test.txt')
Tlabel = np.loadtxt('trainlabel.txt')
Testlb = np.loadtxt('testlabel.txt')

SP0 = Trainset.shape[0]
TS0 = Trainset.shape[1]
SP1 = Testset.shape[0]
TS1 = Testset.shape[1]
Trainlabel = np.reshape(Tlabel,(SP0,1))
Testlabel = np.reshape(Testlb,(SP1,1))

#############################

###### Training the Model ######
################################

model_time0 = time.time()

#rows = Trainset.shape[0]
#cols = Trainset.shape[1]
penalty = 0.001
learning_rate = 0.001
EPOCHS = 4096
BATCH_SIZE = 128
actfunc = tf.nn.tanh
regularizer = regularizers.l2(penalty)

###### LSTM pre-preparation ######
##################################
inputs = Input(shape=(TS0,1))
Train = np.reshape(Trainset,(SP0,TS0,1))
Test = np.reshape(Testset,(SP1,TS1,1))

hidden = LSTM(32,return_sequences=False,activation=actfunc,kernel_regularizer=regularizer)(inputs)
prediction = Dense(1)(hidden)

### Print out the dot function during iterations
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


optimizer = tf.keras.optimizers.RMSprop(learning_rate)
model = Model(inputs=inputs, outputs=prediction)
model.summary()
model.compile(optimizer=optimizer, loss='mean_squared_error', 
        metrics=['mean_squared_error','mean_absolute_error'])    
history = model.fit(Train,Trainlabel,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_split=0.0,verbose=0,callbacks=[PrintDot()])


model_time1 = time.time()

###### Write the Current Model ######
#####################################
model.save('test.h5')


###### Plot the Error Evolution ######
######################################

def plot_history(history):

### Plot the main absolute error ###
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  #plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
   #        label = 'Val Error')
  plt.ylim([-0.5,50])
  plt.legend()
  
### Plot the main squared error ###
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  #plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           #label = 'Val Error')
  plt.ylim([-1,100])
  plt.legend()
  plt.show()


plot_history(history)

test_predictions = model.predict(Test)

###### Write Data to File ######
################################

##write_time0 = time.time()

f = open("output.txt","w")
for i in xrange(SP1):
    f.write("%8.6f\n" % (test_predictions[i]))
f.close()

##write_time1 = time.time()

###### Error Calculation ######
###############################
Error = 0.0
for i in xrange(SP1):
    Error += np.absolute(test_predictions[i]-Testlabel[i])

Error = Error/np.sum(np.absolute(Testlabel))
print("The MRE value is", Error)
