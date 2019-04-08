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


########## PART 1: Loading Data ##########
##########################################

Trainset = np.loadtxt('train.txt')
Testset = np.loadtxt('test.txt')
Trainlabel = np.loadtxt('trainlabel.txt')
Trainlabel0 = np.loadtxt('trainlabel0.txt')
Testlabel = np.loadtxt('testlabel.txt');
Scale = np.loadtxt('scale.txt');


###### PART 2: Setting up Basic Parameters ######
#################################################
In_dim = Trainset.shape[1]
Out_dim = Testlabel.shape[1]
In_sp = Trainset.shape[0]
Out_sp = Testlabel.shape[0]


penalty = 0.001
learning_rate = 0.01
EPOCHS = 6500
BATCH_SIZE = 512
actfunc = tf.nn.relu
regularizer = None


########## PART 3: Training the Model ##########
################################################
inputs = Input(shape=(In_dim,))

hidden = Dense(128, activation=actfunc,kernel_regularizer=regularizer)(inputs)
hidden = Dense(64, activation=actfunc,kernel_regularizer=regularizer)(hidden)
hidden = Dense(16, activation=actfunc,kernel_regularizer=regularizer)(hidden)
prediction = Dense(Out_dim)(hidden)

### Print out the dot function during iterations
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

#class PrintDot(keras.callbacks.Callback):
#    def on_epoch_end(self, epoch, logs):
#        if epoch % 100 == 0:
#            print("Epoch:",'epoch'        

optimizer = tf.keras.optimizers.Adam(learning_rate)
model = Model(inputs=inputs, outputs=prediction)
model.summary()
model.compile(optimizer=optimizer, loss='mean_squared_error', 
        metrics=['mean_squared_error','mean_absolute_error'])    
history = model.fit(Trainset,Trainlabel,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_split=0.0,verbose=0,callbacks=[PrintDot()])


########## PART 4: Post-processing ##########
#############################################
model.save('model_load.h5')

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
#  plt.ylim([-0.5,50])
  plt.legend()
  
### Plot the main squared error ###
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  #plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           #label = 'Val Error')
#  plt.ylim([-1,100])
  plt.legend()
  plt.show()

plot_history(history)

test_predictions = model.predict(Testset)


########## PART 5: Writing Data to File ##########
##################################################

## Unfolding the Prediction Values
test_rearranged = np.zeros((Out_sp,Out_dim))

for d in range(Out_dim):
    for s in xrange(Out_sp):
        test_rearranged[s,d] = test_predictions[s,d]*Scale[1,d]+Scale[0,d]

f = open("output.txt","w")
for i in xrange(Out_sp):
    for j in xrange(Out_dim):
        f.write("%10.8f" % (test_rearranged[i,j]))
    f.write("\n")    
f.close()

## Calculate the Mean Relative Error (MRE)
Error = np.zeros((1,Out_dim))
ErrorT = np.zeros((1,In_dim))

for j in xrange(Out_dim):
    for i in xrange(Out_sp):
        Error[0,j] += np.absolute(test_rearranged[i,j]-Testlabel[i,j])
    Error[0,j] = Error[0,j]/np.sum(np.absolute(Testlabel[:,j]))

print("\n")
print("The mean relative error for the test set:")
print(Error)
