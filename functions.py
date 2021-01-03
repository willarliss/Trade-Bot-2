import os
import pickle

import numpy as np
import pandas as pd
import sqlite3 as sql
from tensorflow.keras.models import load_model
from scipy.stats import pearsonr

def fetch_data(table, db='ticker_data.db'):
    
    connection = sql.connect(db)

    try:

        c = connection.cursor()
        c.execute(f"""
            SELECT * 
            FROM {table};
            """)
        df = pd.DataFrame(c.fetchall(), columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    except Exception as e:
        
        print('EXCEPTION', e)

    connection.close()

    return df

def pickle_model(mod, path='model_info'):
    
    memory = mod.memory
    parameters = np.array([
        mod.action_space, mod.state_space, mod.gamma, 1000, mod.batch_size, mod.alpha, mod.alpha_min, mod.alpha_decay,
        ])
         
    try:
        
        with open(os.path.join(path, 'params.pkl'), 'wb') as f:    
            pickle.dump([memory, parameters], f)

        mod.model.save(os.path.join(path, 'network.pb'))
        
    except FileNotFoundError:
        
        os.mkdir(path)
        
        with open(os.path.join(path, 'params.pkl'), 'wb') as f:    
            pickle.dump([memory, parameters], f)

        mod.model.save(os.path.join(path, 'network.pb'))
        
def unpickle_model(mod_class, path='model_info'):
    
    network = load_model(os.path.join(path, 'network.pb'))
    
    with open(os.path.join(path, 'params.pkl'), 'rb') as f:
        
        info = pickle.load(f)
        memory = info[0]
        parameters = info[1]
    
    model = mod_class(*parameters)
    model.model = network
    
    model.memory = memory
    model.is_fit = True
    
    return model
    