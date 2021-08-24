


model = 'NOK_CNN_Conv1D_LSTM_AEM'




from tensorflow.keras.callbacks import EarlyStopping
from OptimizeNN_keras import CNN_Conv1D_LSTM_sMAPE,SymmetricMeanAbsolutePercentageError
from Raw_Data import format_data_2, get_data
from keras_tuner import HyperParameters
import numpy as np
import json
from tensorflow.keras.optimizers import Adam



import plaidml.keras
import os
plaidml.keras.install_backend()
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

with open('hyperparameter_best/params','r+') as file:
    file_data = json.load(file)

#Assign best Hyperparameters
hp = HyperParameters()
for key in file_data[model]:
    if key[:4:] != 'tuner':
        hp.Fixed(key, file_data[model][key])


#Get data


#TODO:Check what hyperparameter optimization of metrics gives for MSE and MAPE

for i in range(20): #Iterate a few times
    data = get_data('NOK')
    x_data, x_days, y_data, y_days =format_data_2(data, input_length=10, input_quantfiers=['close'], output_length=1,offset=1, output_quantifiers=['close'], sliding_window = False)
    x_data, x_days, y_data, y_days = x_data[-len(x_data)//2::], x_days, y_data[-len(x_data)//2::], y_days

    model = CNN_Conv1D_LSTM_sMAPE(hp)

    callback = EarlyStopping(monitor=metric, patience=10)

    model.fit(x_data, y_data, epochs = 10000,validation_split=0.2,shuffle=True,callbacks=[callback])

    path = './trained_models/{}/{}'.format('NOK', 'CNN_Conv1D_LSTM_{}'.format(metric))
    model.save(path)

    #Get new shifted data
    data = get_data('NOK')[5::]
    x_data, x_days, y_data, y_days =format_data_2(data, input_length=10, input_quantfiers=['close'], output_length=1,offset=1, output_quantifiers=['close'], sliding_window = False)
    x_data, x_days, y_data, y_days = x_data[-len(x_data)//2::], x_days, y_data[-len(x_data)//2::], y_days


    y_pred = np.array(model.predict(x_data))
    y_data = np.array(y_data)

    residuals = np.divide(np.absolute(y_pred-y_data), y_data)*100

    mean = np.mean(residuals)

    std = np.std(residuals)

    with open('./trained_models/{}/{}'.format('NOK', 'accuracy'), 'a') as f:
        f.write('CNN_Conv1D_LSTM_{}\n'.format(metric))
        f.write('mean percentage error: {}, std: {}\n'.format(mean, std))