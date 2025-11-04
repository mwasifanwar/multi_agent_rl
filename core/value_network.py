import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class CentralizedValueNetwork(nn.Module):
    def __init__(self, num_agents, state_dim, hidden_dim=256):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        self.agent_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4)
            ) for _ in range(num_agents)
        ])
        
        self.global_network = nn.Sequential(
            nn.Linear((hidden_dim // 4) * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, observations):
        agent_features = []
        
        for i in range(self.num_agents):
            agent_obs = observations[:, i * self.state_dim:(i + 1) * self.state_dim]
            encoded_features = self.agent_encoders[i](agent_obs)
            agent_features.append(encoded_features)
        
        global_state = torch.cat(agent_features, dim=-1)
        value = self.global_network(global_state)
        
        return value

class MixingNetwork(nn.Module):
    def __init__(self, num_agents, state_dim, hidden_dim=128):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, agent_values, global_state):
        batch_size = agent_values.shape[0]
        
        w1 = torch.abs(self.hyper_w1(global_state)).view(-1, self.num_agents, self.hidden_dim)
        w2 = torch.abs(self.hyper_w2(global_state)).view(-1, self.hidden_dim, 1)
        b1 = self.hyper_b1(global_state).view(-1, 1, self.hidden_dim)
        b2 = self.hyper_b2(global_state).view(-1, 1, 1)
        
        hidden = F.elu(torch.bmm(agent_values.unsqueeze(1), w1) + b1)
        total_value = torch.bmm(hidden, w2) + b2
        
        return total_value.squeeze()

class QMIXNetwork(nn.Module):
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim=128, mixing_hidden=32):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.agent_q_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(num_agents)
        ])
        
        self.mixing_network = MixingNetwork(num_agents, state_dim, mixing_hidden)
    
    def forward(self, observations, global_state):
        batch_size = observations.shape[0]
        
        agent_q_values = []
        for i in range(self.num_agents):
            agent_obs = observations[:, i * self.state_dim:(i + 1) * self.state_dim]
            q_values = self.agent_q_networks[i](agent_obs)
            agent_q_values.append(q_values.unsqueeze(1))
        
        agent_q_values = torch.cat(agent_q_values, dim=1)
        
        total_q_value = self.mixing_network(agent_q_values, global_state)
        
        return total_q_value, agent_q_values