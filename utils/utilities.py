import numpy as np
import pandas as pd
import sqlite3 as sql
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.preprocessing import minmax_scale



def live_plot(data_dict, figsize=(15,5), trace='net_worth'):

    clear_output(wait=True)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    
    for label, data in data_dict.items():
        if trace in label:
            ax.plot(data, label=label)
    
    ax.legend(loc='lower left')
    
    return ax



def fetch_data(db_name):
    
    data = {}
    
    conn = sql.connect(db_name)
    c = conn.cursor()
        
    ticker_query = """
        SELECT name
        FROM sqlite_master
        WHERE type='table';
    """
    
    try:
        c.execute(ticker_query)
    except Exception as e:
        conn.close()
        raise e
        
    for t in c.fetchall():
        ticker = t[0]

        tables_query = f"""
            SELECT *
            FROM {ticker};
        """
        
        try:
            data[ticker] = pd.read_sql(tables_query, con=conn, parse_dates='date')
        except Exception as e:
            conn.close()
            raise e
        
    conn.close()
    return data



def validate_data(dataframes):
    
    assert isinstance(dataframes, dict), 'Historical stock data must be dictionary instance'
    
    lens, firsts, lasts = [], [], []
    cols = ('date', 'close', 'volume', 'ma_30', 'ma_5', 'volatil', 'diff', 'diff_ma_5', )
    
    for ticker, df in dataframes.items():
        
        assert isinstance(df, pd.DataFrame), 'Historical data must be dataframe(s)'
        assert df.index.start == 0, 'Dataframe index must start at 0'
        
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.strip()
        
        assert set(df.columns) == set(cols), f'Missing {set(cols)-set(df.columns)} column(s)'
        
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by='date', ascending=True, inplace=True)

        lens.append(len(df))
        firsts.append(df['date'].iloc[0])
        lasts.append(df['date'].iloc[-1])
        
        assert df.isna().sum().sum() == 0, 'Missing values present in historical data'
        
        df = df[[*cols]]
        
    assert len(set(lens)) == 1, 'Lengths of price histories must match'    
    assert len(set(firsts)) == 1, 'Starting dates of price histories must match'
    assert len(set(lasts)) == 1, 'Ending dates of price histories must match'
        
    
     
def get_indices(n_episodes, min_, max_):
    
    x = np.random.randn(int(n_episodes))
    x = minmax_scale(
        np.abs(x)*-1, 
        (min_, max_, ),
    )
    
    return x.astype(int)


    