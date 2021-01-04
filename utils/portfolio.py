import gym
import numpy as np
import pandas as pd



class Portfolio:
    
    def __init__(self, stocks, balance_init=1_000_000, fee=0.001):
        
        assert isinstance(stocks, list) and len(stocks) >= 1
        assert balance_init >= 100
        
        self.stocks = [stock.lower() for stock in stocks]
        self.balance_init = balance_init
        self.fee = fee
        
    def reset(self):
        
        positions = {pos: 0.0 for pos in self.stocks}
        self.positions_full = positions.copy() # Number of shares held for each stock
        self.positions_norm = positions.copy() # Portfolio exposure of each stock
        self.positions_norm['_out'] = 1.0 # Portfolio exposure includes none

        self.balance = self.balance_init # Liquidity in account
        self.net_worth = self.balance_init # liquidity + shares*prices
        
        self.days_passed = 0
        self.profits = []
        
        return self
        
    def make_trade(self, actions, prices):
        
        assert isinstance(actions, dict)
        assert isinstance(prices, dict)
        
        net_worth_prev = self.net_worth
        
        try:
            out = abs(actions.pop('_out'))
        except KeyError:
            out = 0
        
        sales = {stock: action for stock, action in actions.items() if action < 0}
        purchases = {stock: action for stock, action in actions.items() if action > 0}
        
        # Execute sales first
        for stock, action in sales.items():
            
            # How many shares are held
            total_possible = self.positions_full[stock]
            # Sell the specified portion of available held
            shares_sold = total_possible * -action
            # Profit is the price times quantity minus fee
            profit = shares_sold * prices[stock] * (1 - self.fee)

            self.positions_full[stock] -= shares_sold
            self.balance += profit
                
        # Adjust purchase allocations if necessary   
        if sum(purchases.values()) > 1:
            purchases = {
                k: v/sum(purchases.values())
                for k, v in purchases.items()
            }
  
        # Set aside available balance
        balance = self.balance - (self.balance*out)
        
        # Exectes purchases
        for stock, action in purchases.items():    
            
            # How many shares can be afforded
            total_possible = balance / (prices[stock]*(1+self.fee))
            # Buy specified amount of available shares
            shares_bought = total_possible * action
            # Cost is th eprices times the quantity plus fee
            cost = shares_bought * prices[stock] * (1 + self.fee)
            
            self.positions_full[stock] += shares_bought
            self.balance -= cost

        # Calculate net_worth
        self.net_worth = self.balance + sum(
            shares*price for shares, price 
            in zip(self.positions_full.values(), prices.values())
        )
        
        # Calculate exposures
        for position in self.positions_norm.keys():
            if position == '_out':
                self.positions_norm[position] = self.balance / self.net_worth
            else:
                self.positions_norm[position] = (self.positions_full[position]*prices[position]) / self.net_worth
                
        self.days_passed += 1
        self.profits.append(self.net_worth-net_worth_prev)
        
    def report(self):
        
        print('Balance:', round(self.balance, 5))
        print('Net worth:', self.net_worth)
        print('Shares held:', self.positions_full)
        print('Exposures:', self.positions_norm, '|', round(sum(self.positions_norm.values()), 5))
        
        try:
            print('Current profit:', self.profits[-1])
        except IndexError:
            print("Current profit: NA")
       
        try:
            print('Average profit:', sum(self.profits)/self.days_passed)
        except ZeroDivisionError:
            print("Average profit: NA")
       
        print('Total profit:', sum(self.profits))
        print('n Steps:', self.days_passed)
            
        return sum(self.profits)
    
    
    