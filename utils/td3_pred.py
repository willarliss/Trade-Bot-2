import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    
class PositiveSoftmaxTanh(nn.Module):
    
    def __init__(self):
        
        super(PositiveSoftmaxTanh, self).__init__()
        
    def forward(self, values_full):
                
        for values in values_full:
            
            values[:-1] = torch.tanh(values[:-1])
            values[-1] = torch.sigmoid(values[-1])

            values[:-1][values[:-1]>0] = (
                values[:-1][values[:-1]>0] / torch.sum(values[:-1][values[:-1]>0])
            )

        return values_full
        
        

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
    
        super(Actor, self).__init__()
        
        h1, h2 = 400, 300
        
        self.layer_1 = nn.Linear(state_dim, h1) 
        self.layer_2 = nn.Linear(h1, h2) 
        self.layer_3 = nn.Linear(h2, action_dim)

        self.pst = PositiveSoftmaxTanh()

    def forward(self, state):
                        
        X = F.relu(self.layer_1(state))
        X = F.relu(self.layer_2(X))
        X = self.pst(self.layer_3(X)) 
        
        return X 



class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):

        super(Critic, self).__init__()
        
        h1, h2 = 400, 300

        self.layer_1 = nn.Linear(state_dim + action_dim, h1)
        self.layer_2 = nn.Linear(h1, h2)
        self.layer_3 = nn.Linear(h2, 1)

    def forward(self, state, action):

        Xu = torch.cat([state, action], axis=1)

        X = F.relu(self.layer_1(Xu))
        X = F.relu(self.layer_2(X))
        X = self.layer_3(X)

        return X
    
    
    
class Predictor(nn.Module):
    
    def __init__(self, input_len, output_len, eta=0.01):
        
        super(Predictor, self).__init__()
        
        h1, h2 = 100, 50
        
        self.layer_1 = nn.Linear(input_len, h1)
        self.layer_2 = nn.Linear(h1, h2)
        self.layer_3 = nn.Linear(h2, output_len)
                
    def forward(self, X):
        
        X = F.relu(self.layer_1(X))
        X = F.relu(self.layer_2(X))
        X = self.layer_3(X)
        
        return X



class ReplayBuffer:

    def __init__(self, max_len=1e6):

        self.ptr = 0
        self.storage = []
        self.max_len = max_len
        self.curr_len = len(self.storage)
    
    def add(self, transition):

        if self.curr_len == self.max_len:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_len

        else:
            self.storage.append(transition)
            
        self.curr_len = len(self.storage)

    def sample(self, batch_size):

        idx = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []

        for i in idx:
            state, next_state, action, reward, done = self.storage[i]
            
#            state = state.reshape(-1,1)
#            next_state = next_state.reshape(-1,1)
      
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))

        return (
            np.array(batch_states).squeeze(),
            np.array(batch_next_states).squeeze(),
            np.array(batch_actions),
            np.array(batch_rewards).reshape(-1,1),
            np.array(batch_dones).reshape(-1,1), 
        )
    
    
    
class Agent:

    def __init__(self, state_dim, action_dim, max_action, mem_size=1e6, eta=1e-3):
        
        self.eta = eta
        self.mem_size = mem_size
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.eta)

        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_1_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.eta)

        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.eta)

        self.replay_buffer = self.reset_replay_buffer()
        self.predictor_fit = False

    def select_action(self, state):

        state = torch.FloatTensor(state.reshape(1,-1)).to(DEVICE)
        state = torch.cat([state, self.predictor(state)], axis=1)
        
        return (self.actor(state)
            .cpu()
            .data.numpy()
            .flatten()
        )
    
    def train(self, iterations=100, batch_size=100, gamma=0.99, tau=0.001, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for i in np.arange(iterations):

            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = self.replay_buffer.sample(batch_size)
                        
            X = torch.FloatTensor([state[:-1, :-1].flatten() for state in batch_states])
            y = torch.FloatTensor([state[:-1, 5] for state in batch_next_states])
            self._fit_predictor(X, y)

            batch_states = [state.reshape(-1,1) for state in batch_states]
            batch_next_states = [state.reshape(-1,1) for state in batch_next_states]
            
            state = torch.FloatTensor(batch_states).to(DEVICE)
            next_state = torch.FloatTensor(batch_next_states).to(DEVICE)
            action = torch.FloatTensor(batch_actions).to(DEVICE)
            reward = torch.FloatTensor(batch_rewards).to(DEVICE)
            done = torch.FloatTensor(batch_dones).to(DEVICE)

            state = torch.cat([state, self.predictor(state)], axis=1)
            next_state = torch.cat([next_state, self.predictor(next_state)], axis=1)

            noise = torch.FloatTensor(batch_actions).data.normal_(0, policy_noise)
            noise = noise.clamp(-noise_clip, noise_clip) 

            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)

            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1-done) * gamma * target_Q).detach()       

            current_Q1 = self.critic_1(state, action)
            current_Q2 = self.critic_2(state, action)

            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()

            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()

            if i % policy_freq == 0:

                actor_loss = -self.critic_1(
                    state, 
                    self.actor(state)
                    ).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(
                        (tau * param.data) + ((1-tau) * target_param.data)
                    )

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(
                        (tau * param.data) + ((1-tau) * target_param.data)
                    )

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(
                        (tau * param.data) + ((1-tau) * target_param.data)
                    )
                
    def _fit_predictor(self, X, y, epochs=10):
        
        if not self.predictor_fit:
            self.predictor = Predictor(X.shape[1], y.shape[1])
            self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.eta)
            self.predictor_fit = True
            
        for i in np.arange(epochs):
            _X = self.predictor(X)
            loss = F.mse_loss(_X, y)
            self.predictor_optimizer.zero_grad()
            loss.backward()
            self.predictor_optimizer.step()
            
    def reset_replay_buffer(self, inplace=False, size=None):
        
        if size is None:
            size = self.mem_size
            
        if inplace:
            self.replay_buffer = ReplayBuffer(max_len=size)
        else:
            return ReplayBuffer(max_len=size)
        
    def save(self, filename, directory):

        torch.save(self.actor.state_dict(), f'{directory}/{filename}_actor.pth')
        torch.save(self.critic_1.state_dict(), f'{directory}/{filename}_critic_1.pth')
        torch.save(self.critic_2.state_dict(), f'{directory}/{filename}_critic_2.pth')
        
    def load(self, filename, directory):
    
        self.actor.load_state_dict(torch.load(f'{directory}/{filename}_actor.pth'))
        self.critic_1.load_state_dict(torch.load(f'{directory}/{filename}_critic_1.pth'))
        self.critic_2.load_state_dict(torch.load(f'{directory}/{filename}_critic_2.pth'))



    
    

        
        
        
class RobustAgent(Agent):
    
    def __init__(self, state_dim, action_dim, max_action, mem_size=1e6, eta=1e-3):
        
        super(RobustAgent, self).__init__(
            state_dim, action_dim, max_action, mem_size, eta,
        )
        
        self.predictor_fit = False
        self.eta = eta
        

            
            
    def select_action(self, state):

        state = torch.FloatTensor(state.reshape(1,-1)).to(DEVICE)
        
        return (self.actor(state)
            .cpu()
            .data.numpy()
            .flatten()
        )
    
    def train(self, iterations=100, batch_size=100, gamma=0.99, tau=0.001, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for i in np.arange(iterations):

            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = self.replay_buffer.sample(batch_size)
            
            state = torch.FloatTensor(batch_states).to(DEVICE)
            next_state = torch.FloatTensor(batch_next_states).to(DEVICE)
            action = torch.FloatTensor(batch_actions).to(DEVICE)
            reward = torch.FloatTensor(batch_rewards).to(DEVICE)
            done = torch.FloatTensor(batch_dones).to(DEVICE)

            noise = torch.FloatTensor(batch_actions).data.normal_(0, policy_noise)
            noise = noise.clamp(-noise_clip, noise_clip) 

            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)

            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1-done) * gamma * target_Q).detach()       

            current_Q1 = self.critic_1(state, action)
            current_Q2 = self.critic_2(state, action)

            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()

            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()

            if i % policy_freq == 0:

                actor_loss = -self.critic_1(
                    state, 
                    self.actor(state)
                    ).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(
                        (tau * param.data) + ((1-tau) * target_param.data)
                    )

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(
                        (tau * param.data) + ((1-tau) * target_param.data)
                    )

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(
                        (tau * param.data) + ((1-tau) * target_param.data)
                    )