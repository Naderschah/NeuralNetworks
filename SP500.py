#!/home/felix/anaconda3/bin/python

from Raw_Data import get_data_yf, prep_all_data, ModDataset
import requests
import pandas as pd
import numpy as np
import os
'''
#Getting the list of sp500 stocks
url = 'https://www.liberatedstocktrader.com/sp-500-companies/'
html = requests.get(url).content
df_list = pd.read_html(html)
df = df_list[-2].iloc[1::,1]
#Get data
#for i in df:
#    get_data_yf(i)
#Create directory for new data
if not os.path.isdir('/home/felix/MLP_stockdata'):
    os.mkdir('/home/felix/MLP_stockdata')
#Prep all data for MLP
prep_all_data('/home/felix/stockdata', '/home/felix/MLP_stockdata')
#Oversample all data'''
dataset = ModDataset(dataset_path='/home/felix/MLP_stockdata/MLP_feature_select_dat.csv', preped=False)
#print('Oversampling data')
'''x_train, x_test, y_train, y_test = dataset.oversample()
#TODO: Fix SMOTENC returning killed - RAM issue
for key,i  in zip(('x_train_under', 'x_test_under', 'y_train_under', 'y_test_under'),(x_train, x_test, y_train, y_test)):
    np.savetxt(os.path.join('/home/felix/MLP_stockdata',key+'csv' ), i, delimiter=',')
del x_train, x_test, y_train, y_test'''

print('Undersampling data')
x_train, x_test, y_train, y_test = dataset.undersample()

for key,i  in zip(('x_train_under', 'x_test_under', 'y_train_under', 'y_test_under'),(x_train, x_test, y_train, y_test)):
    np.savetxt(os.path.join('/home/felix/MLP_stockdata',key+'.csv' ), i, delimiter=',')