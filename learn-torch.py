import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
#haut => 1
#bas => 2
#droite => 4
#gauche => 3

class GrivEnv():
    def __init__(self):
        self.state = (0,0)
        self.goal = (4,4)
        self.reward = 0
        self.griv = [[0 for _ in range(5)] for _ in range(5)] 

    def update_reward(self,point):
        self.reward += point
        return self.reward

    def step(self, action):
        x, y = self.state
        if action == 0:
            new_state = (x - 1, y)
        elif action == 1:
            new_state = (x + 1, y)
        elif action == 2:
            new_state = (x, y - 1)
        elif action == 3:
            new_state = (x, y + 1)

        if new_state in ((1,0),(1,1),(1,3),(1,4),(2,3),(2,4),(3,1),(3,3),(3,4)):    
            return self.state, -5, False

        if 0 <= new_state[0] <= 4 and 0 <= new_state[1] <= 4:
            self.state = new_state
            if self.state == self.goal:
                return self.state, 10, True 
            return self.state, -1, False 
        else:
            return self.state, -5, False 
        
    def reset(self):
        self.state = (0, 0)
        self.reward = 0
        return self.state
    
    def griv_render(self):
        for l in self.griv:
            print(l)

    def play(self):
        done = False
        self.reward = 0
        while not done:
            self.griv = [[0,0,0,0,0], 
                        [1,1,0,1,1],
                        [0,0,0,1,1],
                        [0,1,0,1,1],
                        [0,0,0,0,0]]
            action = random.randint(1,4)
            next_state, reward, done = self.step(action)
            self.griv[self.state[0]][self.state[1]] = 1
            print(self.state)
            self.griv_render()
            self.update_reward(reward)
            print()
        print("You win")
        print("Your score is : ",self.reward)
        
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x        

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)
    
# Hyperparamètres
epsilon = 1.0 # choisi un équilibre entre exploration (découvrir de nouvelles actions) et exploitation (utiliser les actions apprises)
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.99   # à quelle point les récompenses futures sont importantes
lr = 0.0005     # taux d'apprentissage
batch_size = 32 # taille du batch pour l'apprentissage
replay_buffer_capacity = 10000 # mémoire du replay buffer
update_target_every = 100  # fréquence de mise à jour du modèle cible
episode = 600

model = DQN() 
optimizer = optim.Adam(model.parameters(), lr=lr)
replay_buffer = ReplayBuffer(replay_buffer_capacity)

env  = GrivEnv()

criterion = nn.MSELoss()
 
def choose_action(state,epsilon):
    if random.random() < epsilon:
        return random.randint(0,3)
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_values = model(state_tensor)
        return torch.argmax(q_values).item() 

for i in range(episode):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = choose_action(state, epsilon)
        
        next_state, reward, done = env.step(action)
        
        replay_buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        
        total_reward += reward
        
        if replay_buffer.size() > batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states_tensor = torch.tensor(states, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.long)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
            dones_tensor = torch.tensor(dones, dtype=torch.float32)
            
            q_values = model(states_tensor)
            q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            
            next_q_values = model(next_states_tensor)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards_tensor + (gamma * next_q_value * (1 - dones_tensor))
            
            loss = criterion(q_value, target_q_value)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if i % 100 == 0:
        print(f"Episode {i}, Total Reward: {total_reward}")

def choose_action_test(state):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    q_values = model(state_tensor)
    return torch.argmax(q_values).item()

model.eval()
with torch.no_grad():
    done = False
    state = env.reset()
    total_reward = 0
    while not done:
        griv = [[0,0,0,0,0],
                [1,1,0,1,1],
                [0,0,0,1,1],
                [0,1,0,1,1],
                [0,0,0,0,0]]
        action = choose_action_test(state)
        next_state, reward, done = env.step(action)
        griv[next_state[0]][next_state[1]] = 5
        for l in griv:
            print(l)
        print(f"Action choisie : {action}, état suivant : {next_state}, récompense : {reward}")
        
        # Mise à jour
        state = next_state
        total_reward += reward
    
    print("Fin :)")
    print("Récompense totale :", total_reward)