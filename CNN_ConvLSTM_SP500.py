#!/home/felix/anaconda3/bin/python

print('\nEmptying GPU memory\n')
from numba import cuda
cuda.select_device(0)
cuda.close()

print('Preping GPU for use\n')
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    print(device)
    tf.config.experimental.set_memory_growth(device, True)

import numpy as np
from keras_tuner import Hyperband, HyperParameters
from OptimizeNN_keras import CNN_Conv1D_LSTM
from Raw_Data import ModDataset, format_data
import os


# Here we will train one CNN_Conv1D_LSTM network for each SP500 stock, as there are a lot of Hyperparameters in this 
# network, we will only be training Networks based on one stock hyperparameters

# The produced networks will predict the next days closing price, the idea is that we will be able to use this data
# to create extra data to be fed to the MLP as time series data i.e. [[todays things][tomorrows][...]]
# however this implementation will require experimentation with regards to appropriate identifiers to use 
# (some need other parameters than closing price)

# Get random data file from SP500 stocks
for root, dirs, files in os.walk('/home/felix/stockdata'):
    file = files[np.random.randint(0,len(files))]

data = ModDataset(dataset_path='/home/felix/stockdata/{}'.format(file))

data = data.data.to_numpy()
#Create datasets of 5 input days of 5 quantifiers, and 1 output day with 1 closing price
x_data, x_days, y_data, y_days = format_data(data,input_length=5,input_quantfiers=['open','high','low','close','volume'],output_length=1,output_quantifiers=['close'],offset=0)

print('Creating model tuner')
tuner = Hyperband(
    hypermodel = CNN_Conv1D_LSTM,
    max_epochs=1000, 
    objective="MAPE", 
    factor=5,
    hyperband_iterations=2, #In my interest to set as high as possible, max_epochs * (math.log(max_epochs, factor) ** 2)  one iteration runs prior nr of epochs across trials
    directory='CNN_ConvLSTM/{}'.format(file.split('_')[1].split('.')[0]),
    project_name= 'hyper_for_generalization'
)
x_data = np.array(x_data)
x_days = np.array(x_days)
y_data = np.array(y_data)
y_days = np.array(y_days)
print('Starting search')
tuner.search(x_data,y_data,validation_split=0.3,shuffle=False,batch_size=128)
print('best parameters: \n',tuner.get_best_hyperparameters()[0].get_config()['values'])

#Fix Hyperparameters
params = tuner.get_best_hyperparameters()[0].get_config()['values']

hp = HyperParameters()
for key in params:
    hp.Fixed(name=key, value=params[key])

#Figure out file handling
if not os.path.isdir('/home/felix/CNN_ConvLSTM'):
    os.mkdir('/home/felix/CNN_ConvLSTM')

del tuner
print('Starting training loop')
for root, dirs, files in os.walk('/home/felix/stockdata'):
    for i in files:
        print(i.split('_')[1].split('.')[0])
        del data, x_data, x_days, y_data, y_days
        #Getting data
        data = ModDataset(dataset_path='/home/felix/stockdata/{}'.format(file))
        data = data.data.to_numpy()
        #Create datasets of 5 input days of 5 quantifiers, and 1 output day with 1 closing price
        x_data, x_days, y_data, y_days = format_data(data,input_length=5,input_quantfiers=['open','high','low','close','volume'],output_length=1,output_quantifiers=['close'],offset=0)
        #Preparing the model
        model = CNN_Conv1D_LSTM(hp)
        #training
        model.fit(x_data,y_data,validation_split=0.3,shuffle=False, batch_size=128)
        #Saving
        model.save(os.path.join('/home/felix/CNN_ConvLSTM', i.split('_')[1].split('.')[0]))
        del model