import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class MAPPO:
    def __init__(self, num_agents, state_dim, action_dim, actor_lr=0.001, critic_lr=0.001, gamma=0.99, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        from core.policy_network import ActorCriticNetwork
        from core.value_network import CentralizedValueNetwork
        
        self.actor_critics = {}
        self.critics = {}
        self.actor_optimizers = {}
        self.critic_optimizers = {}
        
        for i in range(num_agents):
            agent_id = f'agent_{i}'
            
            actor_critic = ActorCriticNetwork(state_dim, action_dim)
            centralized_critic = CentralizedValueNetwork(num_agents, state_dim)
            
            actor_optimizer = optim.Adam(actor_critic.parameters(), lr=actor_lr)
            critic_optimizer = optim.Adam(centralized_critic.parameters(), lr=critic_lr)
            
            self.actor_critics[agent_id] = actor_critic
            self.critics[agent_id] = centralized_critic
            self.actor_optimizers[agent_id] = actor_optimizer
            self.critic_optimizers[agent_id] = critic_optimizer
    
    def select_actions(self, observations: Dict[str, np.ndarray], training=True) -> Tuple[Dict[str, int], Dict[str, Any], Dict[str, Any]]:
        actions = {}
        log_probs = {}
        values = {}
        
        for agent_id, observation in observations.items():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            
            action_probs, value = self.actor_critics[agent_id](obs_tensor)
            
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
    
    def update(self, batch_data: Dict[str, Any], ppo_epochs=4, mini_batch_size=32):
        observations = batch_data['observations']
        actions = batch_data['actions']
        rewards = batch_data['rewards']
        next_observations = batch_data['next_observations']
        dones = batch_data['dones']
        old_log_probs = batch_data['log_probs']
        old_values = batch_data['values']
        
        batch_size = len(observations[list(observations.keys())[0]])
        
        for _ in range(ppo_epochs):
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mini_batch = {}
                
                for key in ['observations', 'actions', 'rewards', 'next_observations', 'dones']:
                    mini_batch[key] = {}
                    for agent_id in batch_data[key].keys():
                        mini_batch[key][agent_id] = batch_data[key][agent_id][start:end]
                
                mini_batch['log_probs'] = {}
                mini_batch['values'] = {}
                for agent_id in old_log_probs.keys():
                    mini_batch['log_probs'][agent_id] = old_log_probs[agent_id][start:end]
                    mini_batch['values'][agent_id] = old_values[agent_id][start:end]
                
                for agent_id in self.actor_critics.keys():
                    if agent_id not in observations:
                        continue
                    
                    self._update_agent(mini_batch, agent_id)
    
    def _update_agent(self, batch_data: Dict[str, Any], agent_id: str):
        observations = batch_data['observations'][agent_id]
        actions = batch_data['actions'][agent_id]
        rewards = batch_data['rewards'][agent_id]
        next_observations = batch_data['next_observations'][agent_id]
        dones = batch_data['dones'][agent_id]
        old_log_probs = batch_data['log_probs'][agent_id]
        old_values = batch_data['values'][agent_id]
        
        obs_tensor = torch.FloatTensor(observations)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_obs_tensor = torch.FloatTensor(next_observations)
        dones_tensor = torch.BoolTensor(dones)
        old_log_probs_tensor = torch.stack(old_log_probs)
        old_values_tensor = torch.stack(old_values)
        
        centralized_obs = self._prepare_centralized_input(batch_data['observations'])
        next_centralized_obs = self._prepare_centralized_input(batch_data['next_observations'])
        
        with torch.no_grad():
            next_values = self.critics[agent_id](next_centralized_obs).squeeze()
            advantages = rewards_tensor + self.gamma * next_values * (~dones_tensor).float() - old_values_tensor.squeeze()
        
        action_probs, values = self.actor_critics[agent_id](obs_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions_tensor)
        entropy = action_dist.entropy().mean()
        
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        current_values = self.critics[agent_id](centralized_obs).squeeze()
        value_loss = nn.MSELoss()(current_values, rewards_tensor + self.gamma * next_values * (~dones_tensor).float())
        
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        self.actor_optimizers[agent_id].zero_grad()
        self.critic_optimizers[agent_id].zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critics[agent_id].parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), 0.5)
        self.actor_optimizers[agent_id].step()
        self.critic_optimizers[agent_id].step()
    
    def _prepare_centralized_input(self, observations: Dict[str, np.ndarray]) -> torch.Tensor:
        batch_size = len(next(iter(observations.values())))
        centralized_obs = []
        
        for i in range(batch_size):
            episode_obs = []
            for agent_id in sorted(observations.keys()):
                episode_obs.extend(observations[agent_id][i])
            centralized_obs.append(episode_obs)
        
        return torch.FloatTensor(centralized_obs)
    
    def save_models(self, filepath: str):
        models = {}
        for agent_id in self.actor_critics.keys():
            models[agent_id] = {
                'actor_critic_state_dict': self.actor_critics[agent_id].state_dict(),
                'critic_state_dict': self.critics[agent_id].state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizers[agent_id].state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizers[agent_id].state_dict()
            }
        torch.save(models, filepath)
    
    def load_models(self, filepath: str):
        models = torch.load(filepath)
        for agent_id, model_data in models.items():
            if agent_id in self.actor_critics:
                self.actor_critics[agent_id].load_state_dict(model_data['actor_critic_state_dict'])
                self.critics[agent_id].load_state_dict(model_data['critic_state_dict'])
                self.actor_optimizers[agent_id].load_state_dict(model_data['actor_optimizer_state_dict'])
                self.critic_optimizers[agent_id].load_state_dict(model_data['critic_optimizer_state_dict'])