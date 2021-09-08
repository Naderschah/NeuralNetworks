
import json
import requests
import csv
import os
import pandas as pd
import numpy as np
import datetime as dt
from scipy.signal.filter_design import cheb1ap
import talib as ta
from scipy.signal import find_peaks
from scipy.optimize import minimize
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.generic_utils import to_snake_case
from scipy.sparse import coo_matrix
from itertools import compress
import time

def write_json(new_data, filename='data.json'):
    with open(filename,'r+') as file:
          # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data.append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)



### Importing data from AlphaVantage

def AV_data(stock,full=True):
    """
    Retrieve Data from Alpha Vantage,
    -------------
    stock -> str: stock ticker
    full -> bool: wheter or not to return complete dataset
    """
    f =  open(os.path.join('/home/felix/keys'),'r')
    content = f.readlines()
    f.close()
    for i in range(len(content)):
        if '[AV]' in content[i]:
            API_KEY = content[i+1].strip('\n')
        
    if full:
        outputsize = 'full'
    else:
        outputsize = 'compact'
    while True:
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize={}&apikey={}&datatype=csv'.format(stock,outputsize, API_KEY)
        res= requests.get(url)
        decoded_content = res.content.decode('utf-8')
        data= list(csv.reader(decoded_content.splitlines(), delimiter=','))[1::][::-1] #Flip list so time series is sequentially forward
        if 'Alpha Vantage' in data[1][0]: #Should contain the message of API key overuse
            time.sleep(15) #Wait 15 seconds
        else:
            break
    return data

import yfinance as yf
def yfinance(stock):
    stock = yf.Ticker(stock)
    data = stock.history(period='max',interval='1d') #I need date open high low close volume
    data.drop(columns=["Dividends", "Stock Splits"], inplace=True)
    print(type(data.iloc[0,0]))
    data.insert(loc=0,column='date',value=data.index.map(lambda x: x))
    return data.to_records(index=True)#.tolist()

if __name__=='__main__':
    print(yfinance('AAPL'))

#Saving Data if not already saved

def get_data(stock): 
    """
    Use as primary data retrieval!
    ----------------
    Collects most recent or all data and saves, assures that all data exists
    """
    #Make directory if not present
    directory = os.path.abspath('../stockdata/')
    if not os.path.isdir(directory): 
        os.mkdir(directory)
    #If the file exists check if data needs to be appended
    if os.path.isfile(os.path.join(directory,stock+'.csv')): 
        newfile = False
        with open(os.path.join(directory,stock+'.csv'),'r', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|') # Get csv file reader
            spamreader =list(spamreader)
            try:
                final = spamreader[-1][0] 
            except: #Index errer
                final = None
            if final != dt.date.today().strftime('%Y-%m-%d'):
                try: 
                    dat = AV_data(stock,full=newfile)
                except: 
                    print('Could not retrieve data, using previously saved data') 
                    return np.array(spamreader) #TODO: Check this works!
            
                for i in range(len(dat)):
                    if dat[i][0] == final: #Get last index of date written to file
                        break
            else:
                return np.array(spamreader)
    else:
        newfile=True
        i = -1 #Write full data set
        try:
            dat = AV_data(stock,full=newfile)
            with open(os.path.join(directory,stock+'.csv'),'a', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL) #Get Writer for csv file
                for j in range(i+1,len(dat)): #Write the data to file
                    spamwriter.writerow(dat[j])
        except Exception as e: 
            print('Could not retrieve data ', e)
            return 0 

    with open(os.path.join(directory,stock+'.csv'),'r', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        return np.array(list(spamreader),dtype=object)
    


def format_data(data, input_length, input_quantfiers, output_length,offset, output_quantifiers, sliding_window = False, scaling_func = None):
    """
    Function to generalize data formating for future usage, always groups into sequence of sequence (i.e. (None, 5,5); (None, 1,1) --> 5 days of 5 quantifiers in and 1 day of one quantifier out)
    ----------
    data --> np.array ---> raw data as returned by AV_data or get_data (preference on get_data as it doesnt use as many API calls)
    input_length --> length of input sequence (i.e. days) in terms of shape (none, x,y) would be x
    input_quantifiers --> list of str: quantifiers to use i.e. ['close', 'open',...]
    output_length --> length of output sequence (i.e. days) see above
    offset --> offset between input and output sequence in days
    output_quantifiers --> list of str: quantifiers to use i.e. ['close', 'open',...]
    sliding_window --> wheter to use sliding window approach or not #FIXME: implement
    scaling_func ---> callable scaling function can also be list of callables in order in which they are to be applied
    -------------
    always returns: x_data, x_days, y_data, y_days, args
    Where args is a dict object containing the output of scaling_func etc.
    """
    #Set dtype to correctly format array for further use
    data = np.array([data.transpose()[0].astype('U10'), *data.transpose()[1::].astype('float64')],dtype=object).transpose()
    quantifier_keys = {'open':1, 'high':2, 'low':3, 'close':4, 'volume':5} #FIXME: Add keys with correct indices
    if 'offset' not in locals():
        offset = 0
    input_indeces = [0]
    
    args = {}
    #   Get keys of quantifiers to include for in and out
    for i in input_quantfiers:
        for j in quantifier_keys:
            if i == j:
                input_indeces.append(quantifier_keys[i]) 
                break

    output_indeces = [0]

    for j in output_quantifiers:
        for i in quantifier_keys:
            if i == j:
                output_indeces.append(quantifier_keys[i])
                break

    #   Apply scaling
    if scaling_func!=None:
        args['order'] = [] #Contains order of scaling operation 
        if type(scaling_func) == list:
            for func in scaling_func:
                data, args = func(data) 
        else:
            data, args = scaling_func(data, args)

    #   Get relevant data
    data = data[-100::].transpose()
    #FIXME: Sort input indeces! Check if np array assignment worked otherwise use np.delete
    x_data = np.array([data[i] for i in input_indeces]).transpose()
    y_data = np.array([data[i] for i in output_indeces]).transpose()
    if not sliding_window:
        nr_of_sets = len(x_data)//(input_length+output_length) 
        if offset > 0: #TODO: CHange the way I accounted for offset
            nr_of_sets-=1
        #Below removes from front rather than end
        x_data = x_data[len(x_data)-(input_length+output_length)*nr_of_sets::]
        y_data = y_data[len(y_data)-(input_length+output_length)*nr_of_sets::]
        if output_length == 1:
            y_days = y_data[input_length-1::input_length+output_length+offset,0]
            y_data = y_data[input_length-1::input_length+output_length+offset,1::] #TODO: Check this works
        else:
            raise Exception('longer than 1 output length not implemented yet!')
            #Could reshape array using numpy and select only relevant indeces
            #Work this out on paper

        x_days = np.delete(x_data, np.linspace(1,len(x_data),len(x_data),dtype=int)[input_length-1::input_length+output_length],axis=0)[::,0]
        x_data = np.delete(x_data, np.linspace(1,len(x_data),len(x_data),dtype=int)[input_length-1::input_length+output_length],axis=0)[::,1::]
        x_data = np.reshape(x_data, (nr_of_sets, input_length, x_data.shape[-1])) #TODO: Check this actualy works


    return x_data, x_days, y_data, y_days, args

    #else: #TODO: Sliding window
    #    raise Exception('Sliding window not implemented yet!')

    

def format_data_2(data, input_length, input_quantfiers, output_length,offset, output_quantifiers, sliding_window = False):
    data = np.array([data.transpose()[0].astype('U10'), *data.transpose()[1::].astype('float64')],dtype=object).transpose()
    quantifier_keys = {'open':1, 'high':2, 'low':3, 'close':4, 'volume':5} #FIXME: Add keys with correct indices
    if 'offset' not in locals():
        offset = 0
    indeces = {}
    args = {}

    input_indeces = []

    #   Get keys of quantifiers to include for in and out
    for i in input_quantfiers:
        for j in quantifier_keys:
            if i == j:
                indeces[quantifier_keys[i]]=1 
                input_indeces.append(quantifier_keys[i]) 
                break

    output_indeces = []

    for j in output_quantifiers:
        for i in quantifier_keys:
            if i == j:
                indeces[quantifier_keys[i]] = 1
                output_indeces.append(quantifier_keys[i])
                break
    #Select only relevant indecies
    data = data.transpose()
    dates = data[0]
    data = np.array([data[key] for key in indeces]).transpose()
    #Number of sets if no data is sourced twice
    nr_of_sets = len(data)//(offset+input_length+output_length)
    #Change length of array
    data = data[len(data)-nr_of_sets*(offset+input_length+output_length)::]
    dates = dates[len(dates)-nr_of_sets*(offset+input_length+output_length)::]
    #Reshape array
    data = np.reshape(data,(nr_of_sets, offset+input_length+output_length, data.shape[-1]))
    dates = np.reshape(dates,(nr_of_sets, offset+input_length+output_length, data.shape[-1]))
    if output_length == 1:
        y_data = np.array([data[i][-1] for i in range(len(data))],dtype=np.float64)
        y_days = np.array([dates[i][-1] for i in range(len(data))],dtype=object)

    else:
        y_data = np.array([data[i][-output_length:] for i in range(len(data))],dtype=np.float64)
        y_days = np.array([dates[i][-output_length:] for i in range(len(data))],dtype=object)

    x_data = np.array([data[i][:-output_length-offset] for i in range(len(data))],dtype=np.float64)
    x_days = np.array([dates[i][:-output_length-offset] for i in range(len(data))],dtype=object)

    return x_data, x_days, y_data, y_days


def prep_all_data(directory,save_files):
    """Pass all downloaded stock data to ModDataset and prep MLP"""
    for dirpath, dirnames, filenames in os.walk(directory):
        for i in filenames:
            filepath = os.path.join(dirpath,i)
            dataset = ModDataset(filepath)
            dataset.data
            dataset.prep_MultiLayerPerceptron(offset=0)
            dataset.save_data(os.path.join(save_files,i[:-4:]))


def append_csv(data,filename):
    """Append to a .pkl file"""
    import numpy as np        
    f=open(filename+'.csv','ab')
    np.savetxt(f,data, delimiter=',', fmt='%d')
    f.close()


class ModDataset:
    """Loads Dataset from path, allows for preparation for MLP"""
    #https://github.com/CanerIrfanoglu/medium/tree/master/candle_stick_recognition
    candle_rankings = {
        "CDL3LINESTRIKE_Bull": 1,
        "CDL3LINESTRIKE_Bear": 2,
        "CDL3BLACKCROWS_Bull": 3,
        "CDL3BLACKCROWS_Bear": 3,
        "CDLEVENINGSTAR_Bull": 4,
        "CDLEVENINGSTAR_Bear": 4,
        "CDLTASUKIGAP_Bull": 5,
        "CDLTASUKIGAP_Bear": 5,
        "CDLINVERTEDHAMMER_Bull": 6,
        "CDLINVERTEDHAMMER_Bear": 6,
        "CDLMATCHINGLOW_Bull": 7,
        "CDLMATCHINGLOW_Bear": 7,
        "CDLABANDONEDBABY_Bull": 8,
        "CDLABANDONEDBABY_Bear": 8,
        "CDLBREAKAWAY_Bull": 10,
        "CDLBREAKAWAY_Bear": 10,
        "CDLMORNINGSTAR_Bull": 12,
        "CDLMORNINGSTAR_Bear": 12,
        "CDLPIERCING_Bull": 13,
        "CDLPIERCING_Bear": 13,
        "CDLSTICKSANDWICH_Bull": 14,
        "CDLSTICKSANDWICH_Bear": 14,
        "CDLTHRUSTING_Bull": 15,
        "CDLTHRUSTING_Bear": 15,
        "CDLINNECK_Bull": 17,
        "CDLINNECK_Bear": 17,
        "CDL3INSIDE_Bull": 20,
        "CDL3INSIDE_Bear": 56,
        "CDLHOMINGPIGEON_Bull": 21,
        "CDLHOMINGPIGEON_Bear": 21,
        "CDLDARKCLOUDCOVER_Bull": 22,
        "CDLDARKCLOUDCOVER_Bear": 22,
        "CDLIDENTICAL3CROWS_Bull": 24,
        "CDLIDENTICAL3CROWS_Bear": 24,
        "CDLMORNINGDOJISTAR_Bull": 25,
        "CDLMORNINGDOJISTAR_Bear": 25,
        "CDLXSIDEGAP3METHODS_Bull": 27,
        "CDLXSIDEGAP3METHODS_Bear": 26,
        "CDLTRISTAR_Bull": 28,
        "CDLTRISTAR_Bear": 76,
        "CDLGAPSIDESIDEWHITE_Bull": 46,
        "CDLGAPSIDESIDEWHITE_Bear": 29,
        "CDLEVENINGDOJISTAR_Bull": 30,
        "CDLEVENINGDOJISTAR_Bear": 30,
        "CDL3WHITESOLDIERS_Bull": 32,
        "CDL3WHITESOLDIERS_Bear": 32,
        "CDLONNECK_Bull": 33,
        "CDLONNECK_Bear": 33,
        "CDL3OUTSIDE_Bull": 34,
        "CDL3OUTSIDE_Bear": 39,
        "CDLRICKSHAWMAN_Bull": 35,
        "CDLRICKSHAWMAN_Bear": 35,
        "CDLSEPARATINGLINES_Bull": 36,
        "CDLSEPARATINGLINES_Bear": 40,
        "CDLLONGLEGGEDDOJI_Bull": 37,
        "CDLLONGLEGGEDDOJI_Bear": 37,
        "CDLHARAMI_Bull": 38,
        "CDLHARAMI_Bear": 72,
        "CDLLADDERBOTTOM_Bull": 41,
        "CDLLADDERBOTTOM_Bear": 41,
        "CDLCLOSINGMARUBOZU_Bull": 70,
        "CDLCLOSINGMARUBOZU_Bear": 43,
        "CDLTAKURI_Bull": 47,
        "CDLTAKURI_Bear": 47,
        "CDLDOJISTAR_Bull": 49,
        "CDLDOJISTAR_Bear": 51,
        "CDLHARAMICROSS_Bull": 50,
        "CDLHARAMICROSS_Bear": 80,
        "CDLADVANCEBLOCK_Bull": 54,
        "CDLADVANCEBLOCK_Bear": 54,
        "CDLSHOOTINGSTAR_Bull": 55,
        "CDLSHOOTINGSTAR_Bear": 55,
        "CDLMARUBOZU_Bull": 71,
        "CDLMARUBOZU_Bear": 57,
        "CDLUNIQUE3RIVER_Bull": 60,
        "CDLUNIQUE3RIVER_Bear": 60,
        "CDL2CROWS_Bull": 61,
        "CDL2CROWS_Bear": 61,
        "CDLBELTHOLD_Bull": 62,
        "CDLBELTHOLD_Bear": 63,
        "CDLHAMMER_Bull": 65,
        "CDLHAMMER_Bear": 65,
        "CDLHIGHWAVE_Bull": 67,
        "CDLHIGHWAVE_Bear": 67,
        "CDLSPINNINGTOP_Bull": 69,
        "CDLSPINNINGTOP_Bear": 73,
        "CDLUPSIDEGAP2CROWS_Bull": 74,
        "CDLUPSIDEGAP2CROWS_Bear": 74,
        "CDLGRAVESTONEDOJI_Bull": 77,
        "CDLGRAVESTONEDOJI_Bear": 77,
        "CDLHIKKAKEMOD_Bull": 82,
        "CDLHIKKAKEMOD_Bear": 81,
        "CDLHIKKAKE_Bull": 85,
        "CDLHIKKAKE_Bear": 83,
        "CDLENGULFING_Bull": 84,
        "CDLENGULFING_Bear": 91,
        "CDLMATHOLD_Bull": 86,
        "CDLMATHOLD_Bear": 86,
        "CDLHANGINGMAN_Bull": 87,
        "CDLHANGINGMAN_Bear": 87,
        "CDLRISEFALL3METHODS_Bull": 94,
        "CDLRISEFALL3METHODS_Bear": 89,
        "CDLKICKING_Bull": 96,
        "CDLKICKING_Bear": 102,
        "CDLDRAGONFLYDOJI_Bull": 98,
        "CDLDRAGONFLYDOJI_Bear": 98,
        "CDLCONCEALBABYSWALL_Bull": 101,
        "CDLCONCEALBABYSWALL_Bear": 101,
        "CDL3STARSINSOUTH_Bull": 103,
        "CDL3STARSINSOUTH_Bear": 103,
        "CDLDOJI_Bull": 104,
        "CDLDOJI_Bear": 104
    }
    
    def __init__(self,dataset_path, preped=False):
        if dataset_path[-4::] =='.csv':
            self.load_csv(dataset_path)
        elif dataset_path[-4::] =='.pkl':
            self.load_pkl(dataset_path)
        elif preped:
            for dirpath, dirnames, filenames in os.walk(dataset_path):
                for i in filenames:
                    if hasattr(self,'data'):
                        self.data=self.data.append(pd.read_pickle(os.path.join(dirpath,i)))
                    else:
                        self.data=self.data = pd.read_pickle(os.path.join(dirpath,i))



    def load_csv(self,dataset_path):
        """Load Individual Dataset"""
        self.data = pd.read_csv(dataset_path, parse_dates=True, sep=' ', names=['Date', 'open', 'high', 'low', 'close', 'volume'])

    def load_pkl(self,dataset_path):
        self.data = pd.read_pickle(dataset_path)

        
    def prep_MultiLayerPerceptron(self,offset=0):
        self.append_MACD()
        self.append_RSI()
        self.append_WilliamsR()
        self.compute_patterns()
        self.data = self.recognize_candlesticks(self.data)
        self.change_to_candlestick()
        self.data = self.data[offset:]
        self.get_optimized_labels(offset=0)
        

    def compute_patterns(self):
        """Check all patterns in TA Lib, positive is a bullish pattern whereas 0 is a bearish pattern"""
        exclude_items = ('CDLCOUNTERATTACK',
                        'CDLLONGLINE',
                        'CDLSHORTLINE',
                        'CDLSTALLEDPATTERN',
                        'CDLKICKINGBYLENGTH')
        candle_names = ta.get_function_groups()['Pattern Recognition']
        for candle in candle_names:
            if candle not in exclude_items:
                self.data[candle] = getattr(ta, candle)(self.data['open'].to_numpy(),self.data['high'].to_numpy(),self.data['low'].to_numpy(),self.data['close'].to_numpy())
        #return ta.CDLABANDONEDBABY(self.data['open'].to_numpy(),self.data['high'].to_numpy(),self.data['low'].to_numpy(),self.data['close'].to_numpy())
        self.data = self.recognize_candlesticks(self.data)

    def recognize_candlesticks(self,df):#https://github.com/CanerIrfanoglu/medium/tree/master/candle_stick_recognition
        """
        Recognizes candlestick patterns and appends 2 additional columns to df;
        1st - Best Performance candlestick pattern matched by www.thepatternsite.com
        2nd - # of matched patterns
        """
        op = df['open'].astype(float)
        hi = df['high'].astype(float)
        lo = df['low'].astype(float)
        cl = df['close'].astype(float)

        candle_names = ta.get_function_groups()['Pattern Recognition']

        # patterns not found in the patternsite.com
        exclude_items = ('CDLCOUNTERATTACK',
                        'CDLLONGLINE',
                        'CDLSHORTLINE',
                        'CDLSTALLEDPATTERN',
                        'CDLKICKINGBYLENGTH')

        candle_names = [candle for candle in candle_names if candle not in exclude_items]


        # create columns for each candle
        for candle in candle_names:
            # below is same as;
            # df["CDL3LINESTRIKE"] = talib.CDL3LINESTRIKE(op, hi, lo, cl)
            df[candle] = getattr(ta, candle)(op, hi, lo, cl)


        df['candlestick_pattern'] = np.nan
        df['candlestick_match_count'] = np.nan
        for index, row in df.iterrows():

            # no pattern found
            if len(row[candle_names]) - sum(row[candle_names] == 0) == 0:
                df.loc[index,'candlestick_pattern'] = "NO_PATTERN"
                df.loc[index, 'candlestick_match_count'] = 0
            # single pattern found
            elif len(row[candle_names]) - sum(row[candle_names] == 0) == 1:
                # bull pattern 100 or 200
                if any(row[candle_names].values > 0):
                    pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bull'
                    df.loc[index, 'candlestick_pattern'] = pattern
                    df.loc[index, 'candlestick_match_count'] = 1
                # bear pattern -100 or -200
                else:
                    pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bear'
                    df.loc[index, 'candlestick_pattern'] = pattern
                    df.loc[index, 'candlestick_match_count'] = 1
            # multiple patterns matched -- select best performance
            else:
                # filter out pattern names from bool list of values
                patterns = list(compress(row[candle_names].keys(), row[candle_names].values != 0))
                container = []
                for pattern in patterns:
                    if row[pattern] > 0:
                        container.append(pattern + '_Bull')
                    else:
                        container.append(pattern + '_Bear')
                rank_list = [self.candle_rankings[p] for p in container]
                if len(rank_list) == len(container):
                    rank_index_best = rank_list.index(min(rank_list))
                    df.loc[index, 'candlestick_pattern'] = container[rank_index_best]
                    df.loc[index, 'candlestick_match_count'] = len(container)
        # clean up candle columns
        cols_to_drop = candle_names# + list(exclude_items)
        df.drop(cols_to_drop, axis = 1, inplace = True)

        return df
    
    def change_to_candlestick(self):
        """Change candlestick row to index"""
        candle_names = ta.get_function_groups()['Pattern Recognition']

        # patterns not found in the patternsite.com
        exclude_items = ('CDLCOUNTERATTACK',
                        'CDLLONGLINE',
                        'CDLSHORTLINE',
                        'CDLSTALLEDPATTERN',
                        'CDLKICKINGBYLENGTH')

        candle_names = [[candle+'_Bear', candle+'_Bull'] for candle in candle_names if candle not in exclude_items]
        candle_names = [cell for row in candle_names for cell in row]
        candle_names = {candle:index+1 for index,candle in enumerate(candle_names)}
        candle_names['NO_PATTERN'] = 0
        self.data['candlestick_pattern'] = self.data['candlestick_pattern'].map(candle_names)


    def prep_for_nn(self):
        """Change candlestick names to integers for training"""
    
    def get_data_MLP(self):
        keys =[]
        for i in self.data.columns:
            if i not in ['Date', 'open', 'high', 'low', 'close', 'volume', 'labels_buy', 'labels_hold', 'labels_sell']:
                keys.append(i)
        dat = self.data.loc[::,keys]
        return dat.to_numpy()

    def get_labels(self):
        dat = self.data.loc[::,['labels_buy','labels_hold','labels_sell']]
        return dat.to_numpy()

    def oversample(self,split=0.2,random_state=12):
        """Randomly Oversample minority class
        To avoid random imbalance between test and validation set we split prior"""
        b = [1,0,0]
        h = [0,1,0]
        s = [0,0,1]
        data = self.data.dropna().iloc[::,6:].to_numpy(dtype='float')
        #Change to last 1/5 is val rest train
        train, test = train_test_split(data,test_size=split, random_state=random_state)
        del data
        x_train = train[::,:-3:] #from index 6 as Date	open	high	low	close	volume comes first
        y_train = train[::,-3::].astype('int')
        x_test = test[::,:-3:] 
        y_test = test[::,-3::].astype('int')
        del train, test
        y_train_onehot = [np.sum(i*[0,1,2]) for i in y_train]
        x_train, y_train = SMOTENC(categorical_features=[self.data.dtypes==object]).fit_resample(x_train, y_train_onehot)
        y_train = np.array([b if i==0 else h if i==1 else s if i==2 else 0 for i in y_train])

        y_test_onehot = [np.sum(i*[0,1,2]) for i in y_test]
        x_test, y_test = SMOTENC(categorical_features=[self.data.dtypes==object]).fit_resample(x_test, y_test_onehot)
        y_test = np.array([b if i==0 else h if i==1 else s for i in y_test])
        del y_test_onehot, y_train_onehot,
        print('After Overfitting')
        print('Test Data Buy: {}, Hold:{}, Sell: {}'.format(*np.sum(y_test,axis=0)))
        print('Train Data Buy: {}, Hold:{}, Sell: {}'.format(*np.sum(y_train,axis=0)))

        return x_train, x_test, y_train, y_test


    def save_data(self,path,rem_NAN = True):
        if rem_NAN:
            self.data = self.data.dropna()
        self.data.to_pickle(path)

    def append_MACD(self,fastperiod=12, slowperiod=26, signalperiod=9):
        macd, macdsignal, macdhist = ta.MACD(self.data['close'].to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        self.data.loc[::,'MACD'] = macd

    def append_WilliamsR(self,period = 14):
        self.data.loc[::,'wr_{}'.format(period)] = self.get_wr(self.data['high'], self.data['low'],self.data['close'], period)
        #self.data = self.data.dropna()

    def append_RSI(self,period= 14):
        delta = self.data['close'].diff()
        up, down = delta.clip(lower=0), delta.clip(upper=0)
        roll_up1 = up.ewm(span=period).mean()
        roll_down1 = down.abs().ewm(span=period).mean()
        RS1 = roll_up1 / roll_down1
        RSI1 = 100.0 - (100.0 / (1.0 + RS1))
        roll_up2 = up.rolling(period).mean()
        roll_down2 = down.abs().rolling(period).mean()
        RS2 = roll_up2 / roll_down2
        RSI2 = 100.0 - (100.0 / (1.0 + RS2))
        self.data.loc[::,'RSI_EMA_{}'.format(period)] = RSI1
        self.data.loc[::,'RSI_SMA_{}'.format(period)] = RSI2

    def get_wr(self,high, low, close, lookback):
        highh = high.rolling(lookback).max() 
        lowl = low.rolling(lookback).min()
        wr = -100 * ((highh - close) / (highh - lowl))
        return wr

    def get_peaks(self,prominence=0.5,rel_height=2, distance=None, optimize=False):
        """Find peaks and throths, however many different versions of this have to be found 
        """
        x=self.data['close'].to_numpy()
        index_max = find_peaks(x,prominence=prominence,rel_height=rel_height,distance=distance)
        index_min = find_peaks(x*-1,prominence=prominence,rel_height=rel_height,distance=distance)
        val_min = x[index_min[0]]
        val_max = x[index_max[0]]
        #Now we convert it into vector format readable by the nn, s.t. [0,0,1] is a sell signal [1,0,0] is a buy, and [0,1,0] is a hold
        labels = np.zeros((len(x),3))
        labels[::,1] = 1 #Set middle index to 1 for hold
        labels[index_min[0]]=[1,0,0] #buy
        labels[index_max[0]]=[0,0,1] #sell
        print('nr_peak/nr_troth = ', len(val_max)/len(val_min))
        print('sum of buy (-1) and sell (+1) ', sum(labels[::,2])-sum(labels[::,0])) #sum sell - sum buy
        if not optimize:
            self.data.loc[::,'labels_buy'] = labels.transpose()[0]
            self.data.loc[::,'labels_hold'] = labels.transpose()[1]
            self.data.loc[::,'labels_sell'] = labels.transpose()[2]
        else:
            return labels
    
    def get_optimized_labels(self,offset=0):
        res = self.maximize(offset=0)
        self.get_peaks(*res, optimize=False)
        print('With sum difference: ', self.get_diff(offset=offset))

    
    def get_diff(self,labels=None, offset=0):
        """Subtracts all buy signals and adds all sell signals
        Assumes 1:1 ratio of data points in alternating fashion
        """
        if type(labels) == type(None):
            buy = self.data.loc[::,'labels_buy'].to_numpy()[offset:]
            sell = self.data.loc[::,'labels_sell'].to_numpy()[offset:]
        else:
            buy = labels.transpose()[0][offset:]
            sell = labels.transpose()[2][offset:]
        delta = 0
        data = self.data.loc[::,'close'].to_numpy()[offset:]
        for i in range(len(data)):
            if buy[i] == 1:
                delta -= data[i]
            elif sell[i] == 1:
                delta += data[i]
        return delta

    def maximize(self,offset=0):
        """Maximize profits based on find_peaks, i.e. a code that maximizes the profit by buying and selling at scipy's found peaks"""
        param = [0,0,0]
        bnds = ((0,100), (0,100),(1,100))
        x=self.data.loc[::,'close'].to_numpy()
        f = lambda param: 1/self.get_optimizer_diff(data=x,offset=offset,prominence=param[0],rel_height=param[1], distance=param[2])
        param_init = 0.5,2,0
        param = minimize(f,param_init,method='SLSQP', bounds=bnds)
        return param.x

    def get_optimizer_diff(self,data,offset=0,prominence=0.5,rel_height=2, distance=None):
            """Packaging function to remove extra calls
            """
            index_max = find_peaks(data,prominence=prominence,rel_height=rel_height,distance=distance)
            index_min = find_peaks(data*-1,prominence=prominence,rel_height=rel_height,distance=distance)
            labels = np.zeros(len(data))
            labels[index_min[0]]=-1
            labels[index_max[0]]=1
            labels = labels[offset:]
            delta = 0
            data = data[offset:]
            for i in range(len(labels)):
                if labels[i] == -1:
                    delta -= data[i]
                elif labels[i] == 1:
                    delta += data[i]
            return delta
