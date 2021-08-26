#!/home/felix/anaconda3/bin/python3.8

from keras_tuner import Hyperband
from OptimizeNN_keras import MultiLayerPerceptron
from Raw_Data import *
import json


import plaidml.keras
import os
plaidml.keras.install_backend()
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'


stocks = ['NOK'] 


for stock in stocks:
    try:
        #Update data file
        data = get_data(stock)
        #Below for closing data and the like
        '''x_data, x_days, y_data, y_days =format_data_2(data, input_length=50, input_quantfiers=['close'], output_length=1,offset=7, output_quantifiers=['close'], sliding_window = False)
        x_data, x_days, y_data, y_days = x_data[-len(x_data)//2::], x_days, y_data[-len(x_data)//2::], y_days'''
        #Below for MLP:
        directorycsv = os.path.abspath('../stockdata/{}'.format(stock+'.csv'))
        dataset = ModDataset(directorycsv)
        dataset.prep_MultiLayerPerceptron(offset=1000)
        directorypkl = os.path.abspath('../stockdata/{}'.format(stock+'.pkl'))
        dataset.save_data(directorypkl)
        y_data = dataset.get_labels()
        x_data = dataset.get_data_MLP()

        #Finding the best set of parameters and writing it to a file
        tuner = Hyperband(
            hypermodel = MultiLayerPerceptron,
            max_epochs=1000, #TODO: set this to a value slightly higher than the expected epochs to convergence for your largest Mode
            objective="MAPE", 
            factor=5,
            hyperband_iterations=2, #In my interest to set as high as possible, max_epochs * (math.log(max_epochs, factor) ** 2)  one iteration runs prior nr of epochs across trials
            directory=os.path.abspath('../Optimization/'),
            project_name= 'MultiLayerPerceptron_{}'.format(stock),
            overwrite=True
            )
        tuner.search(x_data, y_data,
                    validation_split=0.2,verbose=1,shuffle=False)
        print('Completed Search')
        #Saving best parameters
        with open('hyperparameter_best/params','r+') as file:
            try:
                file_data = json.load(file)
            except:
                file_data = {}
            file_data.update({'{}_{}'.format('MultiLayerPerceptron',stock):tuner.get_best_hyperparameters()[0].get_config()['values']})
            json.dump(file_data,file,indent=4)
        print('Saved hyperparameters')
        print(file_data)

    except Exception as e:
        print('Couln`t compute for ', stock)
        print('Due to: ', e)
