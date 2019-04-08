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
from tensorflow.keras.layers import Input, Dense
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

Llabel = Tlabel.shape[0]
Ltest = Testset.shape[0]
Tlabel.reshape((Llabel,1))

#############################

###### Training the Model ######
################################

model_time0 = time.time()

#rows = Trainset.shape[0]
#cols = Trainset.shape[1]
penalty = 0.001
learning_rate = 0.001
EPOCHS = 4096
BATCH_SIZE = 512

inputs = Input(shape=(2,))

hidden0 = Dense(128, activation=tf.nn.relu)(inputs)
hidden0 = Dense(64, activation=tf.nn.relu)(hidden0)
hidden0 = Dense(32, activation=tf.nn.relu)(hidden0)
prediction = Dense(1)(hidden0)

### Print out the dot function during iterations
"""class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')"""

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print("Epoch:",'epoch')
        


optimizer = tf.keras.optimizers.RMSprop(learning_rate)
model = Model(inputs=inputs, outputs=prediction)
model.summary()
model.compile(optimizer=optimizer, loss='mean_squared_error', 
        metrics=['mean_squared_error','mean_absolute_error'])    
history = model.fit(Trainset,Tlabel,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_split=0.0,verbose=0,callbacks=[PrintDot()])


model_time1 = time.time()

###### Write the Current Model ######
#####################################
model.save('test.h5')


###### Plot the Error Evolution ######
######################################

"""def plot_history(history):

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


plot_history(history)"""

pred0 = model.predict(Testset)

###### Write Data to File ######
################################

##write_time0 = time.time()

f = open("HP.txt","w")
for i in xrange(Ltest):
    f.write("%6.4f\n" % (pred0[i]))
f.close()

##write_time1 = time.time()

