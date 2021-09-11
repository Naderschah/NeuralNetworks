#!/home/felix/anaconda3/bin/python3.8

#Use this https://alphascientist.com/hyperparameter_optimization_with_talos.html
# Automatic image creation and saving and eventually automated maximization by specifyin interesting ranges in a file
# Market dependent or if it works cross dependent, maybe create data for that seperately
import time
start = time.time()
import os
from OptimizeNN_keras import MultiLayerPerceptron
from keras_tuner import Hyperband
from Raw_Data import ModDataset, get_data
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

data = get_data('AAPL')

mod = ModDataset(dataset_path=os.path.join(os.path.abspath('../stockdata/'), 'AAPL.csv'))

mod.prep_MultiLayerPerceptron(offset=100)

x_train, x_test, y_train, y_test = mod.oversample()


tuner = Hyperband(
                hypermodel = MultiLayerPerceptron, #TODO: CHeck with BatchNormalization rather than Layer
                max_epochs=100, #TODO: set this to a value slightly higher than the expected epochs to convergence for your largest Mode
                objective="accuracy", #TODO: Change MAPE to something more descriptive, changed loss but also metric?
                factor=3,
                hyperband_iterations=3, #In my interest to set as high as possible, max_epochs * (math.log(max_epochs, factor) ** 2)  one iteration runs prior nr of epochs across trials
                directory='Optimizer',
                project_name= 'MLP_try',
                overwrite=True
        )
tuner.search(x_train, y_train,
            validation_data=(x_test, y_test),
            )


print(tuner.get_best_hyperparameters()[0])

print('Total execution time: ', time.time()-start)
