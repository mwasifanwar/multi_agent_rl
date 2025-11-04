import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = F.relu(self.layer_norm2(self.fc2(x)))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.shared_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        shared_features = self.shared_network(x)
        
        action_logits = self.policy_head(shared_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        value = self.value_head(shared_features)
        
        return action_probs, value

class CentralizedPolicy(nn.Module):
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
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
            nn.ReLU()
        )
        
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for _ in range(num_agents)
        ])
        
    def forward(self, observations):
        agent_features = []
        
        for i in range(self.num_agents):
            agent_obs = observations[:, i * self.state_dim:(i + 1) * self.state_dim]
            encoded_features = self.agent_encoders[i](agent_obs)
            agent_features.append(encoded_features)
        
        global_state = torch.cat(agent_features, dim=-1)
        global_features = self.global_network(global_state)
        
        action_probs = []
        for i in range(self.num_agents):
            agent_logits = self.policy_heads[i](global_features)
            agent_probs = F.softmax(agent_logits, dim=-1)
            action_probs.append(agent_probs)
        
        return torch.stack(action_probs, dim=1)

class AttentionPolicy(nn.Module):
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.agent_embedding = nn.Linear(state_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(num_agents)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, observations):
        batch_size = observations.shape[0]
        
        agent_embeddings = self.agent_embedding(observations.view(batch_size, self.num_agents, self.state_dim))
        
        attended_embeddings, attention_weights = self.attention(
            agent_embeddings, agent_embeddings, agent_embeddings
        )
        
        attended_embeddings = self.layer_norm(agent_embeddings + attended_embeddings)
        
        action_probs = []
        for i in range(self.num_agents):
            agent_logits = self.policy_heads[i](attended_embeddings[:, i, :])
            agent_probs = F.softmax(agent_logits, dim=-1)
            action_probs.append(agent_probs)
        
        return torch.stack(action_probs, dim=1), attention_weights