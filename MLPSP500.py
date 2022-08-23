#!/home/felix/anaconda3/bin/python

print('\nEmptying GPU memory\n')
from grpc import xds_channel_credentials
from numba import cuda
cuda.select_device(0)
cuda.close()

print('Preping GPU for use\n')
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    print(device)
    tf.config.experimental.set_memory_growth(device, True)

 # FIXME: Optimize all parameters, but learning rate, find the most appropriate (for laaaarge batch sizes) then reiterate using batch sizes of 32 or below with optimization of learning rate (idealy learning_rate inversely scales with batch size )
 # FIXME: Also check if this will be better: https://www.tensorflow.org/guide/profiler
import numpy as np
from keras_tuner import Hyperband
from OptimizeNN_keras import MultiLayerPerceptron
from Raw_Data import normalize
import os









'''
#All can be reloaded using tuner.reload with the same parameters

print('Remember to redirect the output!\n\n\n')
print('Creating Oversample data tuner')
tuner = Hyperband(
    hypermodel = MultiLayerPerceptron,
    max_epochs=1000, #TODO: set this to a value slightly higher than the expected epochs to convergence for your largest Mode
    objective="accuracy", 
    factor=5,
    hyperband_iterations=2, #In my interest to set as high as possible, max_epochs * (math.log(max_epochs, factor) ** 2)  one iteration runs prior nr of epochs across trials
    directory='MLP_SP_over',
    project_name= 'MLP_SP_over'
)
print('Retrieving Data')
x_train_over = np.genfromtxt('/home/felix/MLP_stockdata/x_train_over.csv', delimiter=',')
x_test_over = np.genfromtxt('/home/felix/MLP_stockdata/x_test_over.csv', delimiter=',')
y_train_over = np.genfromtxt('/home/felix/MLP_stockdata/y_train_over.csv', delimiter=',')
y_test_over = np.genfromtxt('/home/felix/MLP_stockdata/y_test_over.csv', delimiter=',')
print('Training data points: ', len(x_train_over))
print('Validation data points: ', len(x_test_over))
tuner.search(x_train_over, y_train_over,validation_data=(x_test_over,y_test_over),shuffle=True,batch_size=32)
print('oversampling parameters',tuner.get_best_hyperparameters()[0],'\n\n\n')
del x_test_over, x_train_over, y_train_over, y_test_over

'''
print('Creating Undersampled model tuner')
tuner = Hyperband(
    hypermodel = MultiLayerPerceptron,
    max_epochs=1000, #TODO: set this to a value slightly higher than the expected epochs to convergence for your largest Mode
    objective="accuracy", 
    factor=5,
    hyperband_iterations=2, #In my interest to set as high as possible, max_epochs * (math.log(max_epochs, factor) ** 2)  one iteration runs prior nr of epochs across trials
    directory='MLP_SP_under',
    project_name= 'MLP_SP_normalized_data'
)
#                     The below doesnt work, best shot is the DataGenerator class above, create ID's and save x and y values in batch size as seperate files named with ID
x_train_under = np.genfromtxt('/home/felix/MLP_stockdata/x_train_under.csv', delimiter=',')#[:700000]
x_test_under = np.genfromtxt('/home/felix/MLP_stockdata/x_test_under.csv', delimiter=',')#[:700000]
y_train_under = np.genfromtxt('/home/felix/MLP_stockdata/y_train_under.csv', delimiter=',')#[:700000]
y_test_under = np.genfromtxt('/home/felix/MLP_stockdata/y_test_under.csv', delimiter=',')#[:700000]
print('Training data points: ', len(x_train_under),len(y_train_under))
print('Validation data points: ', len(x_test_under),len(y_test_under))
#print('Creating Data Generator')
#train_under = csv_feature_generator(x_path='/home/felix/MLP_stockdata/x_train_under.csv',
#                                    y_path='/home/felix/MLP_stockdata/y_train_under.csv',
#                                    batch_size=128)
#x_train=tf.data.Dataset.from_generator(csv_feature_generator, args=('/home/felix/MLP_stockdata/x_train_under.csv','/home/felix/MLP_stockdata/y_train_under.csv', 128),output_types=tf.float32)  
#y_train=tf.data.Dataset.from_generator(csv_data_generator, args=('/home/felix/MLP_stockdata/y_train_under.csv', 128),output_types=float)
#val_under = csv_feature_generator(x_path='/home/felix/MLP_stockdata/x_test_under.csv',
#                                    y_path='/home/felix/MLP_stockdata/y_test_under.csv',
#                                    batch_size=128)
#x_val=tf.data.Dataset.from_generator(csv_feature_generator, args=('/home/felix/MLP_stockdata/x_test_under.csv','/home/felix/MLP_stockdata/y_test_under.csv', 128),output_types=tf.float32) 
#y_val=tf.data.Dataset.from_generator(csv_data_generator, args=('/home/felix/MLP_stockdata/y_test_under.csv', 128),output_types=float)

#tuner.search(x_train,validation_data=x_val,verbose=1,shuffle=True)#,batch_size=128

#Normalize data
print('Normalizing data')
#(x_train_under, constants) = normalize(x_train_under)
#x_test_under = normalize(x_test_under, constants)
#print('Normalization constants: ', constants)
#Save normalization values
#np.savetxt(os.path.join('/home/felix/NeuralNetworks/MLP_SP_under','MLP_SP_normalized_data_normalization_constants'), constants, delimiter=',')
#Start the search
print('Starting search')
tuner.search(x_train_under,y_train_under,validation_data=(x_test_under,y_test_under),shuffle=True,batch_size=128)#
print('undersampling: ',tuner.get_best_hyperparameters()[0])
