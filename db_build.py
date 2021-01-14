import os
import datetime

import sqlite3 as sql
import numpy as np
import pandas as pd

from utils.utilities import fetch_data, validate_data

_links = (
    'https://finance.yahoo.com/quote/FB/history?period1=1337299200&period2=1609632000&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true',
    'https://finance.yahoo.com/quote/AMZN/history?period1=1167782400&period2=1609632000&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true',
    'https://finance.yahoo.com/quote/AAPL/history?period1=1167782400&period2=1609632000&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true',
    'https://finance.yahoo.com/quote/MSFT/history?period1=1167782400&period2=1609632000&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true',
    'https://finance.yahoo.com/quote/GOOG/history?period1=1136246400&period2=1609632000&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true',
)

PATH = './data/'
CUTOFF = datetime.datetime(year=2012, month=7, day=9)
DB_NAME = 'HistoricalPriceData.db'

 
    
def build_table(conn, table_name, table_data):
    
    table_data.to_sql(
        name=table_name, 
        con=conn, 
        index=False, 
        if_exists='replace',
    )
    
    conn.commit()
    

    
def test():
    
    data = fetch_data(
        os.path.join(
            PATH, 
            DB_NAME,
        )
    )
    
    try:
        validate_data(data)
        print('Valid')
    except Exception as e:
        print('ERROR:', e)


        
def main(): 
    
    data = {}
    
    for ticker in ('FB', 'AMZN', 'AAPL', 'MSFT', 'GOOG', ):
    
        df = pd.read_csv(
            os.path.join(PATH, ticker+'.csv')
        )
    
        df.columns = df.columns.str.lower()
    
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by='date', ascending=True, inplace=True)
        
        df['ma_30'] = df['close'].rolling(window=30, center=False).mean()
        df['ma_5'] = df['close'].rolling(window=5, center=False).mean()
        df['volatil'] = np.abs(df['ma_30']-df['close'])
        df['diff'] = df['close'].diff(periods=1)
        df['diff_ma_5'] = df['close'].diff(periods=1).rolling(window=5, center=False).mean()
        
        df = df[df['date'] >= CUTOFF]    
        df.dropna(axis=0, inplace=True)
        df.drop(['open', 'high', 'low', 'adj close'], axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
            
        print(df.shape, df.loc[0, 'date'])
    
        data[ticker.lower()] = df
    
    connection = sql.connect(
        os.path.join(
            PATH, 
            DB_NAME,
        )
    )
    
    for ticker in data:
        
        try:
            build_table(connection, ticker, data[ticker])
            print('Done', ticker, sep=' - ')
    
        except Exception as e:
            print('Error', ticker, e, sep=' - ')
            
    connection.close()
    

    
if __name__ == '__main__':
    main()
    test()
    
    
         