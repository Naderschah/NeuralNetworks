from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten,InputLayer, Dense, LSTM, MaxPooling1D, Reshape, BatchNormalization,Input, LeakyReLU, TimeDistributed, Dropout, Bidirectional, concatenate
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import MeanAbsolutePercentageError, MeanSquaredError
#FIXME:           Important to add an import line for LossFunctionWrapper in __init__ as I could not figure out a direct import
from tensorflow.keras.losses import LossFunctionWrapper
import tensorflow.keras.backend as backend
from keras_tuner import HyperParameters



def CNN_Conv1D_LSTM(hp):
    '''After hyperparameter optimization and seperate training using parameters and EarlyStopping results in 3-5% MAPE and 3-5% std'''
        
    model = Sequential(
                [   #input_shape=(5,5)
                    Dense(units=hp.Int("dense1",min_value=32, max_value=512, step=32),use_bias=hp.Boolean('Bias1')), 
                    LeakyReLU(alpha=hp.Float("alpha1",min_value=0, max_value=1, step=0.11)),
                    Conv1D(filters=hp.Choice('conv1',values=[80,24]), kernel_size=1, use_bias=hp.Boolean('Bias2')),
                    LeakyReLU(alpha=hp.Float("alpha2",min_value=0, max_value=1, step=0.11)),
                    MaxPooling1D(pool_size=2,padding='same',data_format='channels_first'),
                    Conv1D(filters=48, kernel_size=1, use_bias=False),
                    LeakyReLU(alpha=hp.Float("alpha3",min_value=0, max_value=1, step=0.11)),
                    MaxPooling1D(pool_size=2,padding='same',data_format='channels_first'),
                    LeakyReLU(alpha=hp.Float("alpha4",min_value=0, max_value=1, step=0.11)),
                    LSTM(units=hp.Choice('LSTM 1', values=[32,40]),dropout=hp.Float("lstm dropout 1",min_value=0, max_value=1, step=0.11), return_sequences=True,use_bias=hp.Boolean('Bias3'),unit_forget_bias=False),
                    TimeDistributed(Dense(units=hp.Int('dense2', min_value=16,max_value=256,step=32))),
                    LSTM(units=hp.Choice('LSTM 2', values=[16,32]),dropout=hp.Float("lstm dropout 2",min_value=0, max_value=1, step=0.11), use_bias=hp.Boolean('Bias4'),unit_forget_bias=False),
                    Dense(units=hp.Int("dense3",min_value=32, max_value=512, step=32)), 
                    LeakyReLU(alpha=hp.Float("alpha5",min_value=0, max_value=1, step=0.11)),
                    Dense(units=1)
                ]
            )
    
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning rate',values=[0.5,1e-3,1e-4,1e-2,1e-1,1e-5])),
                    loss=SymmetricMeanAbsolutePercentageError(),
                    metrics=['MAPE', 'MSE'])
    
    return model


def CNN_UNIV_10(hp):
    """Generally Performs poor on 10 in 1 out"""
    model = Sequential(
        [   
            Input(shape=(10,1)),
            BatchNormalization(axis=-1),
            Conv1D(filters=hp.Int('conv1',min_value=4,max_value=124, step=12),kernel_size=3, use_bias=False),
            LeakyReLU(alpha=hp.Float("alpha1",min_value=0.1, max_value=1, step=0.1)),
            MaxPool1D(pool_size=hp.Int('MaxPool1',min_value=4, max_value=8,step=2),data_format='channels_last'),
            Flatten(data_format='channels_last'),
            Dense(units=hp.Int('dense1', min_value=25, max_value=55, step=15),use_bias=True),
            LeakyReLU(alpha=hp.Float("alpha2",min_value=0.1, max_value=1, step=0.1)),
            Dense(units=1,use_bias=True),
            LeakyReLU(alpha=hp.Float("alpha3",min_value=0.1, max_value=1, step=0.1))
        ]
    )
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning rate',values=[0.5,0.1,1e-2,1e-3,1e-4,1e-5])),
                            loss=SymmetricMeanAbsolutePercentageError(),
                            metrics=['MAPE'])
    
    return model


def MP_CNN_Bi_Dir_LSTM(hp):
    """
    Still have to Optimize


    Novel Deep Learning Model with CNN and Bi-Directional 
    LSTM for Improved Stock Market Index Prediction - 2019
    -------

    input 50 closing prices
    returns closing price prediction for 7 days into the future
    """
    input1 = Input((50,1))
    normalized = BatchNormalization(axis=-1)(input1)
    model_1 = MP_CNN_Bi_Dir_LSTM_submodel(hp)(normalized)
    model_2 = MP_CNN_Bi_Dir_LSTM_submodel(hp)(normalized)
    model_3 = MP_CNN_Bi_Dir_LSTM_submodel(hp)(normalized)
    model_concat = concatenate([model_1,model_2,model_3], axis = -1)
    model_concat = Dense(1, activation='linear')(model_concat)
    model = Model(inputs=[input1],outputs=model_concat)
    #TODO: Try also using Adadelta optimizer, allthough Adam most likely will perform better
    model.compile(optimizer=Adadelta(learning_rate=hp.Choice('learning rate',values=[0.15,0.1,1e-2]),rho=hp.Float('rho',0.05,0.35,step=0.05), epsilon=hp.Choice('epsilon',values=[1e-3,1e-4,1e-5])),
                            loss=SymmetricMeanAbsolutePercentageError(),
                            metrics=['MAPE'])
    return model

def MP_CNN_Bi_Dir_LSTM_submodel(hp):
    """
    Definition for the three submodels
    """
    Model = Sequential()
    Model.add(Input(shape=(50,1)))
    Model.add(Conv1D(filters=128, kernel_size=hp.Choice('kernel 1',values=[9]))) #TODO: Make a conditional thing, if two kernels are 9 the third has to be 5
    Model.add(LeakyReLU(alpha=hp.Float("alpha1",min_value=0, max_value=1, step=0.11)))
    Model.add(MaxPooling1D(pool_size=2))
    Model.add(Conv1D(filters=128, kernel_size=hp.Choice('kernel 2',values=[9])))
    Model.add(LeakyReLU(alpha=hp.Float("alpha2",min_value=0, max_value=1, step=0.11)))
    Model.add(MaxPooling1D(pool_size=2))
    Model.add(Conv1D(filters=128, kernel_size=hp.Choice('kernel 3',values=[5])))
    Model.add(LeakyReLU(alpha=hp.Float("alpha3",min_value=0, max_value=1, step=0.11)))
    Model.add(MaxPooling1D(pool_size=2))
    Model.add(Bidirectional(LSTM(units=200)))
    Model.add(Dropout(rate=0.5))
    Model.add(Dense(units=1,activation='linear'))
    return Model





def MultiLayerPerceptron(hp):
    """Model to make final Buy Hold and Sell decisions
    Uses Technical Indicators (See if i can predict those with the other networks)
    https://arxiv.org/pdf/1712.09592.pdf
    """
    model = Sequential(
                    [
                        Dense(hp.Int('Dense1',min_value=2,max_value=12, step=2), activation='sigmoid'),
                        Dense(hp.Int('Dense2',min_value=5,max_value=21, step=2), activation='sigmoid'),
                        Dense(hp.Int('Dense3',min_value=2,max_value=12, step=2), activation='sigmoid'),
                        Dropout(0.2),
                        Dense(3, activation='softmax')
                    ]
                       )

    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning rate',values=[0.15,0.1,1e-2]), epsilon=hp.Choice('epsilon',values=[1e-3,1e-4,1e-5])),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
    
    return model









"""
Get fixed Hyperparameters 

"""

def fix_hyperparams(params):
    hp = HyperParameters()
    for i in params:
        hp.Fixed(i, value=params[i])
    return hp
        











"""
Custom Loss functions, if used in hyperparameter tuning place in LossFunctionWrapper, FIXME: Figure out how tho
"""

def smape_loss(y_true, y_pred):
    """sMAPE - symmetric Mean Absolute Percentage Error"""
    epsilon = 0.1
    summ = backend.maximum(backend.abs(y_true) + backend.abs(y_pred) + epsilon, 0.5 + epsilon)
    smape = backend.abs(y_pred - y_true) / summ * 2.0
    return smape

class SymmetricMeanAbsolutePercentageError(LossFunctionWrapper):
    """
    Wrapper for sMAPE loss function, allows to use in hyperparameter tuning
    """
    def __init__(self, name='symmetric_mean_absolute_percentage_error'):
        super(SymmetricMeanAbsolutePercentageError, self).__init__(
            smape_loss, name=name)







