
import json
from numpy.core.fromnumeric import transpose
import requests
import csv
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

### JSON writer

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
    f =  open(os.path.abspath('~/keys'),'r')
    content = f.readlines()
    f.close()
    for i in range(len(content)):
        if '[AV]' in content[i]:
            API_KEY = content[i+1].strip('\n')
        
    if full:
        outputsize = 'full'
    else:
        outputsize = 'compact'
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize={}&apikey={}&datatype=csv'.format(stock,outputsize, API_KEY)
    res= requests.get(url)
    decoded_content = res.content.decode('utf-8')
    return list(csv.reader(decoded_content.splitlines(), delimiter=','))[1::][::-1] #Flip list so time series is sequentially forward

#Saving Data if not already saved

def get_data(stock): 
    """
    Use as primary data retrieval!
    ----------------
    Collects most recent or all data and saves, assures that all data exists
    """
    #Make directory if not present
    directory = os.path.abspath('~/stockdata/')
    if not os.path.isdir(directory): 
        os.mkdir(r'stockdata')
    #If the file exists check if data needs to be appended
    if os.path.isfile(os.path.join(directory,stock+'.csv')): 
        newfile = False
        with open(os.path.join(directory,stock+'.csv'),'r', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|') # Get csv file reader
            spamreader =list(spamreader)
            final = spamreader[-1][0]
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
        except: 
            print('Could not retrieve data, using previously saved data') 
    with open(os.path.join(directory,stock+'.csv'),'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL) #Get Writer for csv file
        for j in range(i+1,len(dat)): #Write the data to file
            spamwriter.writerow(dat[j])
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

