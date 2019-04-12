import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import gym
import numpy as np
import random

'''
STEP 3: CREATE MODEL CLASS
'''
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()
        
        # Linear function 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()

        # Linear function 4 (readout): 100 --> 10
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        # Non-linearity
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)
        
        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 3 (readout)
        out = self.fc3(out)
        # Non-linearity
        out = 2 * self.tanh(out)
        return out

class FeedforwardHybridNeuralNetModel(nn.Module):
    def __init__(self, input_1_dim, input_2_dim, hidden_dim, output_dim):
        super(FeedforwardHybridNeuralNetModel, self).__init__()
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(input_1_dim, hidden_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()
        
        # Linear function 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim + input_2_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()

        # Linear function 3 (readout): 100 --> 10
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        # Non-linearity
        self.tanh = nn.Tanh()
    
    def forward(self, x, y):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)
        
        # Linear function 2
        out = self.fc2(torch.cat((out, y)))
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 3 (readout)
        out = self.fc3(out)
        # Non-linearity
        # out = self.tanh(out)
        return out


'''
STEP 4: INSTANTIATE MODEL CLASS
'''

##### Define the environment
##### Environment : Pendulum-v0
env = gym.make('Pendulum-v0')
state_dim = 3
hidden_dim = 400
action_dim = 1

Q = FeedforwardHybridNeuralNetModel(state_dim, action_dim, hidden_dim, 1)
U = FeedforwardNeuralNetModel(state_dim, hidden_dim, action_dim)

Q_prime = FeedforwardHybridNeuralNetModel(state_dim, action_dim, hidden_dim, 1)
U_prime = FeedforwardNeuralNetModel(state_dim, hidden_dim, action_dim)


#######################
#  USE GPU FOR MODEL  #
#######################


# Check and possibly modify the gpu device used
if torch.cuda.is_available():
    # print(torch.cuda.current_device())
    # torch.cuda.set_device(1)
    pass

if torch.cuda.is_available():
    Q.cuda()
    U.cuda()
    Q_prime.cuda()
    U_prime.cuda()

R = list()

# Discount factor
gamma = 0.9

# Number of epochs
M = 100
T = 100

# Number of samples for computing loss etc.
N = 100

# Total number of (state, action, reward, next_state) tuples to be stored
S = 1000

# Learning rate for optimizing critic
learning_rate_critic = 0.001
learning_rate_actor = 0.0001

# Optimizer for critic
optimizer_critic = torch.optim.Adam(Q.parameters(), lr=learning_rate_critic)

# Optimizer for actor
optimizer_actor = torch.optim.Adam(U.parameters(), lr=learning_rate_actor)

tau = 0.01

# Initialize replay buffers
R_index = 0
R_size = 0
R_state = torch.zeros(0, state_dim)
R_action = torch.zeros(0, action_dim)
R_reward = torch.zeros(0)
R_observation = torch.zeros(0, state_dim)

if torch.cuda.is_available():
    R_state = R_state.cuda()
    R_action = R_action.cuda()
    R_reward = R_reward.cuda()
    R_observation = R_observation.cuda()

for e in range(0, M):

    # Initialize the environment
    state = np.array(env.reset())
    if torch.cuda.is_available():
        state = torch.from_numpy(state).float().cuda()
    else:
        state = torch.from_numpy(state).float()


    for t in range(0, T):

        # Determine the action and obtain a transition by simulation
        if torch.cuda.is_available():
            action = U(state).detach().cuda()
            observation, reward, done, info = env.step(action.cpu().numpy())
            print('Observed transition reward: ' + str(reward))
            reward = torch.from_numpy(np.array([reward])).float().cuda()
            observation = torch.from_numpy(observation).float().cuda()
        else:
            action = U(state).detach()
            observation, reward, done, info = env.step(action.numpy())
            print('Observed transition reward: ' + str(reward))
            reward = torch.from_numpy(np.array([reward])).float()
            observation = torch.from_numpy(observation).float()


        # Store the obtained transition in the replay buffer

        # If buffer not full
        if R_index < S:
            # Append the transition elements to the respective tensors
            R_state = torch.cat((R_state, state.detach().view(1, state_dim)))
            R_action = torch.cat((R_action, action.detach().view(1, action_dim)))
            R_reward = torch.cat((R_reward, reward.view(1, 1)))
            R_observation = torch.cat((R_observation, observation.view(1, state_dim)))
            R_index += 1
            if R_index == M:
                R_index = 0
            R_size += 1
        # Buffer full
        else:
            # Replace transition at index
            R_state[index] = state
            R_action[index] = index
            R_reward[index] = reward
            R_observation[index] = observation
            R_index += 1
            if R_index == M:
                R_index = 0

        # Sample random minibatch of indices of the replay buffer
        if R_size < S:
            pass
        index_list = random.choices(range(0, R_size), k=N)

        optimizer_critic.zero_grad()

        # Calculate loss for the critic
        if torch.cuda.is_available():
            y = torch.cuda.FloatTensor(N)
            q = torch.cuda.FloatTensor(N)
        else:
            y = torch.FloatTensor(N)
            q = torch.FloatTensor(N)

        for i in range(0, len(index_list)):
            ind = index_list[i]

            obs = R_observation[ind].detach()
            obs.requires_grad_()
            action_ = U_prime(obs)
            y[i] = gamma * Q_prime(obs, action_) + R_reward[ind].detach()

            state = R_state[ind].detach()
            state.requires_grad_()
            action = R_action[ind].detach()
            action.requires_grad_()
            q[i] = Q(state, action)

        loss = torch.mean((y - q) ** 2)
        loss.backward()
        optimizer_critic.step()


        # Calculate loss for the actor

        optimizer_actor.zero_grad()
        
        if torch.cuda.is_available():
            q = torch.cuda.FloatTensor(N)
        else:
            q = torch.FloatTensor(N)
        for i in range(0, len(index_list)):
            ind = index_list[i]

            state = R_state[ind].detach()
            state.requires_grad_()
            action = U(state)
            q[i] = Q(state, action)

        J_ = - torch.mean(q)
        J_.backward()
        print('Estimated cost function for the sample : ' + str(J_))
        optimizer_actor.step()
        
        
        # Update secondary networks

        for p, p_prime in zip(Q.parameters(), Q_prime.parameters()):
            p_prime.data = tau * p.data + (1 - tau) * p_prime.data

        for p, p_prime in zip(U.parameters(), U_prime.parameters()):
            p_prime.data = tau * p.data + (1 - tau) * p_prime.data

        if not done:
            state = observation
        else:
            print('Termination state reached')
            state = np.array(env.reset())
            if torch.cuda.is_available():
                state = torch.from_numpy(state).float().cuda()
            else:
                state = torch.from_numpy(state).float()
