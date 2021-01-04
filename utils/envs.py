import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import clear_output

from utils.utilities import validate_data
from utils.portfolio import Portfolio



class BaseTradingEnvironment(gym.Env):
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, stock_data, balance_init=1e6, fee=1e-3):

        self.fee = fee
        self.balance_init = balance_init
        
        self.agent_portfolio = Portfolio(
            list(stock_data.keys()), 
            self.balance_init, 
            self.fee,
        )
        
        self.long_portfolio = Portfolio(
            list(stock_data.keys()), 
            self.balance_init, 
            self.fee,
        )
    
        self.stocks = stock_data.copy()
        validate_data(self.stocks)
        
        self.positions = ( *[t for t in self.stocks.keys()], '_out', )
        self.observations = len(self.stocks[self.positions[0]])
        self.scalers = {t: df.drop('date', axis=1).values[0] for t, df in self.stocks.items()}
        
        self.action_space = spaces.Box(
            low=np.array([ *[-1]*len(self.stocks), 0, ]),
            high=1.0,
            shape=(len(self.stocks)+1, ),
            dtype=np.float64,
        )

        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(self.stocks), 7),
        )

    def _next_observation(self):
        
        full_observation = []
        
        for ticker, df in self.stocks.items():
            
            obs = df.loc[
                self.current_step,
                ['close', 'volume', 'ma_30', 'ma_5', 'volatil', 'diff', 'diff_ma_5'],
            ]
            
            obs /= self.scalers[ticker]
            
            full_observation.append(obs)
            
        return np.array(full_observation).reshape(self.observation_space.shape).astype(np.float64)
    
    def _take_action(self, action):
                
        actions_long = {ticker: 1 for ticker in self.stocks.keys()}
        prices = {ticker: df.loc[self.current_step, 'close'] for ticker, df in self.stocks.items()}
        
        self.agent_portfolio.make_trade(action, prices)
        self.long_portfolio.make_trade(actions_long, prices)
        
        self.net_worth.append(self.agent_portfolio.net_worth)
        self.net_worth_long.append(self.long_portfolio.net_worth)
        
        self.balance = self.agent_portfolio.balance
        self.shares_held = sum(self.agent_portfolio.positions_full.values())
            
    def step(self, action):
        
        assert isinstance(action, dict)
        assert len(action) == self.action_space.shape[0]
        assert '_out' in action.keys()
        
        self._take_action(action)
        self.current_step += 1
        
        done = (
            round(self.balance, 9) < 0
            or self.current_step >= self.observations-1
        )
        
        obs = self._next_observation()
        
        profit_agent = self.agent_portfolio.net_worth - self.agent_portfolio.balance_init
        profit_long = self.long_portfolio.net_worth - self.long_portfolio.balance_init
        reward = (profit_agent - profit_long) / self.balance_init
        
        info = {}
        
        return obs, reward, done, info        
            
    def reset(self):
        
        self.current_step = 1

        self.agent_portfolio.reset()
        self.long_portfolio.reset()
                
        self.balance = self.balance_init
        self.net_worth = [self.balance]
        self.net_worth_long = [self.balance]
        
        return self._next_observation()
        
    def render(self, mode='human', figsize=(16,10), indicator='close'):
        
        clear_output(wait=True)
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2,2)

        ax1 = fig.add_subplot(gs[0,:])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[1,1])

        for ticker, df in self.stocks.items():
            
            date_range = df.loc[1:self.current_step, 'date']
            stock_dta = df.loc[1:self.current_step, indicator]
            
            ax1.plot(
                date_range, 
                stock_dta/df[indicator].std(), 
                label=f'{ticker.upper()} ({indicator})',
            )
                    
        ax2.bar(
            [k if '_' in k else k.upper() for k in self.agent_portfolio.positions_norm.keys()],
            [v for v in self.agent_portfolio.positions_norm.values()],
        )
                    
        ax3.plot(df.loc[1:self.current_step, 'date'], self.net_worth, label='Agent Strategy')
        ax3.plot(df.loc[1:self.current_step, 'date'], self.net_worth_long, label='Long Strategy')
        
        ax1.set_title('Indicator History')
        ax1.legend(loc='upper left')
        
        ax2.set_title('Exposures')
        
        ax3.set_title('Net Worth')
        ax3.legend(loc='upper left')
        
        plt.show()

    @staticmethod
    def format_action(positions, quantities):
        
        pq = zip(positions, quantities)
        
        return dict(pq)        

        
class TradingEnv1(BaseTradingEnvironment):
    
    pass
    
    
    
class TradingEnv2(BaseTradingEnvironment):
    
    def _next_observation(self):
        
        full_observation = []
        
        for ticker, df in self.stocks.items():
            
            obs = df.loc[
                self.current_step,
                ['close', 'volume', 'ma_30', 'ma_5', 'volatil', 'diff', 'diff_ma_5'],
            ]
            
            obs /= self.scalers[ticker]
            
            full_observation.append(obs)
            
        meta = None ###
        full_observation.append(meta)
            
        return np.array(full_observation).reshape(self.observation_space.shape)


        