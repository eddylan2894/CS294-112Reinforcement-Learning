#reference : https://www.tensorflow.org/tutorials/keras/basic_regression

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pickle

num_l = 128

def plot_history(history):
  fig = plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['loss']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_loss']),
           label = 'Val loss')
  plt.legend()
  #plt.ylim([0, 5])
  fig.savefig('Cost.png')

def build_model(in_shape, out_shape ):
  model = keras.Sequential([
    keras.layers.Dense(num_l, activation=tf.nn.relu,
                       input_shape=( in_shape, )),
    keras.layers.Dense(num_l, activation=tf.nn.relu),
    keras.layers.Dense(out_shape)
  ])

  optimizer = keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


#########################################################

EPOCHS = 300

file_name = 'Hopper-v2'


with open('expert_data/'+file_name + '.pkl', 'rb') as f:
        data = pickle.loads(f.read())

data_ob = data['observations']
data_ac = data['actions']
#data_ob = data_ob[:,None,:]
data_ac = data_ac.reshape(data_ac.shape[0],data_ac.shape[2])

model = build_model(data_ob.shape[1], data_ac.shape[1] )
model.summary()


# Store training stats
history = model.fit(data_ob, data_ac, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])
plot_history(history)


model.save(file_name+str(num_l)+'_model.h5')
