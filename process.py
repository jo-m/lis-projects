import numpy as np
import pandas as pd
import datetime as dt
import dateutil

#Daten CSV in pandas dataframe, date ist ein datetime Objekt
def load_data(fname):
    data = pd.read_csv('data/%s.csv' % fname,
                       index_col=False,
                       header=None,
                       names=['date', 'A', 'B', 'C', 'D', 'E', 'F'])
    data['date'] = data['date'].apply(dateutil.parser.parse) 
    return data
    

def featureSelection(data):
    
    data['date'] = data['date'].apply(dateutil.parser.parse)    
    data['weekday0'] = data['date'].apply(dt.datetime.weekday)
    data['weekend'] = data['date'].apply(dt.datetime.weekday) >4
    data['month'] = data['date'].apply(lambda x:x.month)
    data['hour0'] = data['date'].apply(lambda x:x.hour)
    data['year0'] = data['date'].apply(lambda x:x.year)

    del data['date']
    del data['month']
    #del data['weekday0']
    del data['A']
    del data['B']
    del data['C']
    del data['D']
    del data['E']
    del data['F']
    
    gradHour=30
    for i in range(1,gradHour):
        data[ 'hour'+str(i)] = data['hour0']*data['hour'+str(i-1)]
    gradWeekday=3
    for i in range(1,gradWeekday):
        data[ 'weekday'+str(i)] = data['weekday0']*data['weekday'+str(i-1)]
    gradYear=3
    for i in range(1,gradYear):
        data[ 'year'+str(i)] = data['year0']*data['year'+str(i-1)]
    
    for col in data.columns:
        std=np.std(data[col])
        mean=np.mean(data[col])
        data[col]=(data[col]-mean)/std
        
    return data
    
    
    
    