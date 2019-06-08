import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path

class DeepQNetwork(nn.Module):
    def __init__(self, ALPHA, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims


        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        #print(observation)
        #print(self.device)
        state = T.Tensor(observation).to(self.device)
        #observation = observation.view(-1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)
        return actions



class Agent(object):
    def __init__(self, lr, input_dims, gamma=0.99, n_actions=4, lt_size=256, l2_size=256, l3_size = 256):

        self.fileName = "snake.model" #used to save the model

        self.gamma = gamma
        self.reward_memory = []
        self.action_memory = []
        self.policy = DeepQNetwork(lr, input_dims, lt_size, l2_size, l3_size, n_actions)

    def choose_action(self, observation):
        probabilities = F.softmax(self.policy.forward(observation))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()
        #print(self.reward_memory)
        G = np.zeros(len(self.reward_memory), dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean)/std
        G = T.tensor(G, dtype=T.float).to(self.policy.device)
        loss = 0

        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob

        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []


    def saveModel(self):
        T.save({'model_state_dict': self.policy.state_dict(),
                'optimizer_state_dict':self.policy.optimizer.state_dict()}, self.fileName)
    def loadModel(self, lr, input_dims, lt_size, l2_size, l3_size, n_actions):
        rounds = 0
        myfile = Path(self.fileName)
        policy = DeepQNetwork(lr, input_dims, lt_size, l2_size, l3_size, n_actions)
        if myfile.is_file():
            checkpoint = T.load(self.fileName)
            rounds = checkpoint['rounds']
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.policy.optimizer.load_dict(checkpoint['optimizer_state_dict'])
        return(policy, rounds)
