import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Eval_Model():
    def __init__(self,model, stockdata_path):
        self.data ={}
        self.pred ={}
        for dirpath, dirnames, filenames in os.walk(stockdata_path):
            for i in filenames:
                if 'train' not in i and 'test' not in i:
                    try:
                        self.data[i.split('_')[1]]=pd.read_pickle(os.path.join(dirpath,i))
                    except:
                        print('Could not load ', i)
                    #Create datetime objects
                    self.data[i.split('_')[1]]['Date']=self.data[i.split('_')[1]]['Date'].map(lambda x: dt.datetime.strptime(x,'%Y-%m-%dT00:00:00.000000000'))
                    #Create prediction
                    self.pred[i.split('_')[1]]=model.predict(self.data[i.split('_')[1]].iloc[::,6:-3].to_numpy())


    def compute(self):
        self.res = []
        for key in self.data.keys():
            money = 100
            stocks = 0
            for i in range(len(self.pred[key])):
                decision = self.get_index(self.pred[key][i]) #buy Hold sell
                if decision==1: 
                    stocks += money/self.data[key].loc[::,'close'].iloc[i]
                    money = 0
                if decision==2: 
                    pass
                if decision==3: 
                    money += stocks*self.data[key].loc[::,'close'].iloc[i]
                    stocks = 0
            self.res.append([key, (self.data[key].loc[::,'Date'].iloc[-1]-self.data[key].loc[::,'Date'].iloc[0]).total_seconds(), (money+stocks*self.data[key].loc[::,'close'].iloc[-1]-100)])

    def get_index(self,arr):
        for i in range(len(arr)):
            if arr[i] >= 0.5:
                arr[i] = 1
            else:
                arr[i] = 0
        res = np.sum(arr*[1,2,3])
        if res == 0:
            return 1 #Return hold in case of undetermined
        else:
            return res

    def make_plot(self, year = False, day =False):
        fig, ax=plt.subplots(nrows=1, ncols=2,figsize=(32,16))
        plt_data =[]
        for i in self.res:
            if year:
                plt_data.append(i[-1]/(i[-2]/31536000))
            if day:
                plt_data.append(i[-1]/(i[-2]/86400))
        ax[0].set_title('Annualized return ({})'.format(['day' if day else 'year'][0]))
        ax[0].hist(np.array(plt_data), bins=200)
        ax[0].set_xlabel('Percentage Return')
        print('mean ',u"\u00B1", ' std:', np.mean(plt_data), u"\u00B1",np.std(plt_data))
        print('median', np.nanmedian(plt_data))
        print('Ignoring the top 50 results')
        plt_data.sort()
        plt_data=plt_data[:-50]
        ax[1].set_title('Annualized return ignoring top 50 ({})'.format(['day' if day else 'year'][0]))
        ax[1].hist(np.array(plt_data), bins=200)
        ax[1].set_xlabel('Percentage Return')
        print('mean ',u"\u00B1", ' std:', np.mean(plt_data), u"\u00B1",np.std(plt_data))
        print('median', np.nanmedian(plt_data))
        plt.show()

    def buy_and_hold_plot(self):
        fig = plt.figure(figsize=(16,16))
        diff = []
        for key in self.data.keys():
            diff.append((((self.data[key].loc[::,'close'].iloc[-1]-self.data[key].loc[::,'close'].iloc[0])/self.data[key].loc[::,'close'].iloc[0])*100-100)/((self.data[key].loc[::,'Date'].iloc[-1]-self.data[key].loc[::,'Date'].iloc[0]).total_seconds()/31536000))
        plt.title('Buy and Hold (ignoring top 50) per year')
        plt.xlabel('Percentage Return')
        diff.sort()
        plt.hist(diff[:-50], bins=200)
        plt.show()