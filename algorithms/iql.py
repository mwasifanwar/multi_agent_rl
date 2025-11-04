import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class IQL:
    def __init__(self, num_agents, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        from core.policy_network import PolicyNetwork
        from core.value_network import ValueNetwork
        
        self.policies = {}
        self.value_networks = {}
        self.policy_optimizers = {}
        self.value_optimizers = {}
        
        for i in range(num_agents):
            agent_id = f'agent_{i}'
            
            policy = PolicyNetwork(state_dim, action_dim)
            value_net = ValueNetwork(state_dim)
            
            policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
            value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)
            
            self.policies[agent_id] = policy
            self.value_networks[agent_id] = value_net
            self.policy_optimizers[agent_id] = policy_optimizer
            self.value_optimizers[agent_id] = value_optimizer
    
    def select_actions(self, observations: Dict[str, np.ndarray], training=True) -> Dict[str, int]:
        actions = {}
        
        for agent_id, observation in observations.items():
            if training and np.random.random() < self.epsilon:
                actions[agent_id] = np.random.randint(self.action_dim)
            else:
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                action_probs = self.policies[agent_id](obs_tensor)
                actions[agent_id] = torch.argmax(action_probs, dim=-1).item()
        
        return actions
    
    def update(self, batch_data: Dict[str, Any]):
        observations = batch_data['observations']
        actions = batch_data['actions']
        rewards = batch_data['rewards']
        next_observations = batch_data['next_observations']
        dones = batch_data['dones']
        
        for agent_id in self.policies.keys():
            if agent_id not in observations:
                continue
                
            self._update_value_network(agent_id, observations[agent_id], rewards[agent_id], 
                                     next_observations[agent_id], dones[agent_id])
            self._update_policy(agent_id, observations[agent_id], actions[agent_id])
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _update_value_network(self, agent_id, observations, rewards, next_observations, dones):
        value_net = self.value_networks[agent_id]
        
        obs_tensor = torch.FloatTensor(observations)
        rewards_tensor = torch.FloatTensor(rewards)
        next_obs_tensor = torch.FloatTensor(next_observations)
        dones_tensor = torch.BoolTensor(dones)
        
        with torch.no_grad():
            next_values = value_net(next_obs_tensor).squeeze()
            target_values = rewards_tensor + self.gamma * next_values * (~dones_tensor).float()
        
        current_values = value_net(obs_tensor).squeeze()
        value_loss = nn.MSELoss()(current_values, target_values)
        
        self.value_optimizers[agent_id].zero_grad()
        value_loss.backward()
        self.value_optimizers[agent_id].step()
    
    def _update_policy(self, agent_id, observations, actions):
        policy = self.policies[agent_id]
        value_net = self.value_networks[agent_id]
        
        obs_tensor = torch.FloatTensor(observations)
        actions_tensor = torch.LongTensor(actions)
        
        with torch.no_grad():
            state_values = value_net(obs_tensor).squeeze()
        
        action_probs = policy(obs_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions_tensor)
        
        policy_loss = -(log_probs * state_values).mean()
        
        self.policy_optimizers[agent_id].zero_grad()
        policy_loss.backward()
        self.policy_optimizers[agent_id].step()
    
    def save_models(self, filepath: str):
        models = {}
        for agent_id in self.policies.keys():
            models[agent_id] = {
                'policy_state_dict': self.policies[agent_id].state_dict(),
                'value_net_state_dict': self.value_networks[agent_id].state_dict(),
                'policy_optimizer_state_dict': self.policy_optimizers[agent_id].state_dict(),
                'value_optimizer_state_dict': self.value_optimizers[agent_id].state_dict()
            }
        torch.save(models, filepath)
    
    def load_models(self, filepath: str):
        models = torch.load(filepath)
        for agent_id, model_data in models.items():
            if agent_id in self.policies:
                self.policies[agent_id].load_state_dict(model_data['policy_state_dict'])
                self.value_networks[agent_id].load_state_dict(model_data['value_net_state_dict'])
                self.policy_optimizers[agent_id].load_state_dict(model_data['policy_optimizer_state_dict'])
                self.value_optimizers[agent_id].load_state_dict(model_data['value_optimizer_state_dict'])