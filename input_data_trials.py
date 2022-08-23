

import sys
from imblearn.over_sampling._smote import base
from numpy.lib.npyio import save
from scipy.sparse import data
#TSFS testing infrastructure from https://github.com/alimirzaei/TSFS
# In skfeature several functions had to be moved and the main was somewhat modified to allow the chain to work, could not figure out why my_isomap didnt work
sys.path.append('/home/felix/')
#TNFS testing infrastructure from https://github.com/alimirzaei/TNFS
sys.path.append('/home/felix/TNFS/')

import pickle
import os 
from scipy import io
import numpy as np
import os
import talib

from Raw_Data import ModDataset

class InputTrials:
    """Class to be used in varying which input data is suppose to be used"""
    def __init__(self, model=None, base_data=None, save_path=None):
        """model - is the model architecture which is to be checked
        base_data - string to dataset to be used"""
        self.model = model
        if base_data != None:
            #Load the dataset and append all indicators
            if os.path.isdir(base_data):
                print('Loading Datasets in directory')
                self.data=ModDataset(dataset_path=base_data, preped=True)
            else:
                print('Loading Dataset')
                self.data=ModDataset(dataset_path=base_data, preped=False)
            print('Getting talib indicators and retrieving labels')
            self.data.prep_feature_selection(offset=100)
            print('Undersampling data')
            print(self.data.data.iloc[0])
            x_train, x_test, y_train, y_test=self.data.undersample(split=0.01)
            print('Creating matlab data files')
            del self.data
            if save_path == None:
                save_path = '/home/felix/mat_data'
            self.create_mat_dataset(path=save_path+'.mat', x=x_train, y=y_train)
            del x_train, x_test, y_train, y_test
            self.data = save_path+'.mat'
        else:
            self.data = None 

    def do_all(self):
        """Use all available methods to narrow down"""
        raise Exception('To be implemented')

    def create_mat_dataset(self, path, x, y):
        """Creates matlab style datafile"""
        x = np.array(x)
        #Create one hot encoded vector
        y = np.array([i*[1,2,3] for i in y])
        dictionary = {'X':x, 'Y':y}
        io.savemat(path, dictionary)
        self.data=path

    def TSFS(self, datasets=None):
        """Teacher-Student Feature selection as outlined in https://arxiv.org/pdf/1903.07045.pdf"""
        #We will import a modified main, so that we can pass our own dataset, and make choices as to what is to be computed
        # I included all possible tests as I rather approach this using brute force
        from TSFS.test_all import main as TSFS
        print("Starting trials")
        if datasets==None:
            data=self.data
        TSFS(directory='/home/felix/feature_selection/', datasets=[data]) #I cant actually find the test

    def TSNE(self, datasets=None):
        from TSFS.test_all import main as TSFS
        print("Starting trials")
        if datasets==None:
            data=self.data
        TSFS(directory='/home/felix/feature_selection/', datasets=[data]) #I cant actually find the test

        feature = {}
        mean = {}
        std = {}
        for root, dirs, files in os.walk('/home/felix/feature_selection'):
            for i in files:
                with open(os.path.join(root,i),'rb') as filecont:
                    cont=pickle.load(filecont)['/home/felix/MLP_stockdata/feature_selection.mat']
                    feature[i[:-4:]] = cont['feature_ranking']
                    mean[i[:-4:]]=cont['mean'][0]
                    std[i[:-4:]]=cont['std']
        features = feature['my_tsne']
        indicators = [talib.get_function_groups()[i] for i in [i for i in talib.get_function_groups().keys() if i in ('Volume Indicators','Momentum Indicators',)]]
        indicators = [cell for row in indicators for cell in row]
        functions = [indicators[i] for i in range(len(features)) if features[i]<6]
        return functions
    

    def TNFS(self):
        """https://github.com/alimirzaei/TNFS"""
        raise Exception('To be implemented')
