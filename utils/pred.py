import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class Network(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        
        super(Network, self).__init__()
        
        h1, h2 = 200, 100
        
        self.layer_1 = nn.Linear(input_dim, h1) 
        self.layer_2 = nn.Linear(h1, h2) 
        self.layer_3 = nn.Linear(h2, output_dim)
        
    def forward(self, X):
        
        X = F.relu(self.layer_1(X))
        X = F.relu(self.layer_2(X))
        X = self.layer_3(X)
        
        return X
        
    
    
class Predictor:
    
    def __init__(self, input_dim, output_dim, eta=1e-3, alpha=0.0):
                
        self.model = Network(input_dim, output_dim)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=eta,
            weight_decay=alpha,
        )
        
        self.X_full = []
        self.y_full = []
    
    def commit(self, X, y):
        
        self.X_full.append(X)
        self.y_full.append(y)
        
    def sample(self, batch_size=100):

        idx = np.random.randint(0, len(self.X_full), size=batch_size)
        
        X, y = [], []
        for i in idx:
            X.append(self.X_full[i][:-1, :-2].flatten().copy())
            y.append(self.y_full[i][:-1, 5].copy())

        return X, y

    def train(self, epochs=10, batch_size=100):
        
        X, y = self.sample(batch_size)
        
        X = torch.stack([torch.FloatTensor(x_) for x_ in X])
        y = torch.stack([torch.FloatTensor(y_) for y_ in y])
            
        for i in np.arange(epochs):
            
            loss = F.mse_loss(self.model(X), y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return self
            
    def predict(self, X):
        
        _X = torch.FloatTensor(
            X[:-1, :-2].flatten().copy()
        )

        y = self.model(_X)

        return np.c_[X, [*y.data, 0]].astype(np.float64)

    def predict_many(self, X):
        
        X_new = []
        
        for x in X:
            
            X_new.append(
                self.predict(x)
            )
            
        return np.array(X_new)
        
    def save(self, filename, directory):

        torch.save(self.model.state_dict(), f'{directory}/{filename}_predictor.pth')
        
    def load(self, filename, directory):
    
        self.model.load_state_dict(torch.load(f'{directory}/{filename}_predictor.pth'))
        

    