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

import time 

##print(tf.__version__)

############### PART 1: Loading Data ################
#####################################################
Parameters = np.loadtxt('parameters.txt')
Scale = np.loadtxt('scale.txt')

Trainset = np.loadtxt('train.txt')
Testset = np.loadtxt('test.txt')
Tlabel = np.loadtxt('trainlabel.txt')
Testlb = np.loadtxt('testlabel.txt')


######## PART 2: Setting up Basic Parameters ########
#####################################################
TS = int(Parameters[0])     # Number of time steps
DIM = int(Parameters[1])    # Number of dimensions
SPRT = int(Parameters[2])   # The seperation point  
TOTAL = int(Parameters[3])  # The total number of data pieces
SP0 = SPRT-TS+1             # Samples of the training set
SP1 = TOTAL-SPRT-1          # Samples of the test set

Trainlabel = np.reshape(Tlabel,(SP0,DIM))
Testlabel = np.reshape(Testlb,(SP1,DIM))


## PART 3: Setting up the Training-Related Parameters ##
########################################################
penalty = 0.001            # Penalty value for regularization
learning_rate = 0.01       # Learninbg rate - 0.01 as recommended
EPOCHS = 16384             # Learning epochs
BATCH_SIZE = 32            # Size of the mini-batch
actfunc = tf.nn.tanh       # Activation function - tanh for LSTM
#regularizer = regularizers.l2(penalty)
regularizer = None         # No regularization

############# PART 4: Training the Model ############
#####################################################
inputs = Input(shape=(TS,DIM))
Train = np.reshape(Trainset,(SP0,TS,DIM))
Test = np.reshape(Testset,(SP1,TS,DIM))

hidden = LSTM(64,return_sequences=True,activation=actfunc,kernel_regularizer=regularizer)(inputs)
hidden = LSTM(16,return_sequences=False,activation=actfunc,kernel_regularizer=regularizer)(hidden)
#hidden = Dense(32,activation=actfunc,kernel_regularizer=regularizer)(hidden)
hidden = Dense(10,activation=tf.nn.relu,kernel_regularizer=regularizer)(hidden)
prediction = Dense(DIM)(hidden)

## Print Dot Function
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


optimizer = tf.keras.optimizers.Adam(learning_rate)
model = Model(inputs=inputs, outputs=prediction)
model.summary()
model.compile(optimizer=optimizer, loss='mean_squared_error', 
        metrics=['mean_squared_error','mean_absolute_error'])    
history = model.fit(Train,Trainlabel,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_split=0.0,verbose=0,callbacks=[PrintDot()])

############# PART 5: Post-processing #############
###################################################
model.save('saved_model.h5')

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

###### PART 6: Data Writting and Rearranging ######
###################################################

## Unfolding the Prediction Values
test_results = np.reshape(test_predictions,(SP1,DIM))
test_rearranged = np.zeros((SP1,DIM))

for d in range(DIM):
    for s in xrange(SP1):
        test_rearranged[s,d] = test_results[s,d]*Scale[1,d]+Scale[0,d]

f = open("output.txt","w")
for i in xrange(SP1):
    for j in xrange(DIM):
        f.write("%8.6f " % (test_rearranged[i,j]))
    f.write("\n")
f.close()


## Calculate the Mean Relative Error (MRE)
Error = np.zeros((1,DIM))

for j in xrange(DIM):    
    for i in xrange(SP1):
        Error[0,j] += np.absolute(test_rearranged[i,j]-Testlabel[i,j])
    Error[0,j] = Error[0,j]/np.sum(np.absolute(Testlabel[:,j]))

print("\n")
print("The mean relative error for each dimention:")
print(Error)
