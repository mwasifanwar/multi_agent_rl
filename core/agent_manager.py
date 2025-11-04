import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class AgentManager:
    def __init__(self, num_agents, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        self.agents = {}
        self.optimizers = {}
        
        self._initialize_agents()
    
    def _initialize_agents(self):
        from .policy_network import PolicyNetwork
        from .value_network import ValueNetwork
        
        for i in range(self.num_agents):
            agent_id = f'agent_{i}'
            
            policy_net = PolicyNetwork(self.state_dim, self.action_dim)
            value_net = ValueNetwork(self.state_dim)
            
            policy_optimizer = optim.Adam(policy_net.parameters(), lr=self.learning_rate)
            value_optimizer = optim.Adam(value_net.parameters(), lr=self.learning_rate)
            
            self.agents[agent_id] = {
                'policy': policy_net,
                'value': value_net,
                'policy_optimizer': policy_optimizer,
                'value_optimizer': value_optimizer
            }
    
    def select_actions(self, observations: Dict[str, np.ndarray], training=True) -> Dict[str, int]:
        actions = {}
        log_probs = {}
        values = {}
        
        for agent_id, observation in observations.items():
            if agent_id in self.agents:
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                
                policy_net = self.agents[agent_id]['policy']
                value_net = self.agents[agent_id]['value']
                
                action_probs = policy_net(obs_tensor)
                value = value_net(obs_tensor)
                
                if training:
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                else:
                    action = torch.argmax(action_probs, dim=-1)
                    log_prob = torch.log(action_probs[0, action])
                
                actions[agent_id] = action.item()
                log_probs[agent_id] = log_prob
                values[agent_id] = value
        
        return actions, log_probs, values
    
    def update_agents(self, batch_data: Dict[str, Any]):
        observations = batch_data['observations']
        actions = batch_data['actions']
        rewards = batch_data['rewards']
        next_observations = batch_data['next_observations']
        dones = batch_data['dones']
        log_probs = batch_data['log_probs']
        values = batch_data['values']
        
        for agent_id in self.agents.keys():
            if agent_id not in observations:
                continue
                
            policy_loss, value_loss = self._compute_agent_loss(
                agent_id, observations[agent_id], actions[agent_id],
                rewards[agent_id], next_observations[agent_id],
                dones[agent_id], log_probs[agent_id], values[agent_id]
            )
            
            self.agents[agent_id]['policy_optimizer'].zero_grad()
            policy_loss.backward()
            self.agents[agent_id]['policy_optimizer'].step()
            
            self.agents[agent_id]['value_optimizer'].zero_grad()
            value_loss.backward()
            self.agents[agent_id]['value_optimizer'].step()
    
    def _compute_agent_loss(self, agent_id, observations, actions, rewards, next_observations, dones, old_log_probs, old_values):
        observations_tensor = torch.FloatTensor(observations)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_observations_tensor = torch.FloatTensor(next_observations)
        dones_tensor = torch.BoolTensor(dones)
        old_log_probs_tensor = torch.stack(old_log_probs)
        old_values_tensor = torch.stack(old_values)
        
        policy_net = self.agents[agent_id]['policy']
        value_net = self.agents[agent_id]['value']
        
        with torch.no_grad():
            next_values = value_net(next_observations_tensor).squeeze()
            target_values = rewards_tensor + self.gamma * next_values * (~dones_tensor).float()
        
        current_values = value_net(observations_tensor).squeeze()
        value_loss = nn.MSELoss()(current_values, target_values)
        
        advantages = target_values - old_values_tensor.detach()
        
        action_probs = policy_net(observations_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions_tensor)
        
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages).mean()
        
        return policy_loss, value_loss
    
    def get_agent_models(self) -> Dict[str, Any]:
        models = {}
        for agent_id, agent_data in self.agents.items():
            models[agent_id] = {
                'policy_state_dict': agent_data['policy'].state_dict(),
                'value_state_dict': agent_data['value'].state_dict(),
                'policy_optimizer_state_dict': agent_data['policy_optimizer'].state_dict(),
                'value_optimizer_state_dict': agent_data['value_optimizer'].state_dict()
            }
        return models
    
    def load_agent_models(self, models: Dict[str, Any]):
        for agent_id, model_data in models.items():
            if agent_id in self.agents:
                self.agents[agent_id]['policy'].load_state_dict(model_data['policy_state_dict'])
                self.agents[agent_id]['value'].load_state_dict(model_data['value_state_dict'])
                self.agents[agent_id]['policy_optimizer'].load_state_dict(model_data['policy_optimizer_state_dict'])
                self.agents[agent_id]['value_optimizer'].load_state_dict(model_data['value_optimizer_state_dict'])
    
    def compute_cooperative_reward(self, individual_rewards: Dict[str, float]) -> Dict[str, float]:
        total_reward = sum(individual_rewards.values())
        cooperative_rewards = {}
        
        for agent_id in individual_rewards.keys():
            cooperative_rewards[agent_id] = total_reward / len(individual_rewards)
            
        return cooperative_rewards
    
    def compute_competitive_reward(self, individual_rewards: Dict[str, float]) -> Dict[str, float]:
        max_reward = max(individual_rewards.values())
        min_reward = min(individual_rewards.values())
        
        competitive_rewards = {}
        for agent_id, reward in individual_rewards.items():
            if max_reward != min_reward:
                normalized_reward = (reward - min_reward) / (max_reward - min_reward)
            else:
                normalized_reward = 0.5
            competitive_rewards[agent_id] = normalized_reward
            
        return competitive_rewards