import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class MADDPG:
    def __init__(self, num_agents, state_dim, action_dim, actor_lr=0.001, critic_lr=0.001, gamma=0.95, tau=0.01):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        self.actors = {}
        self.critics = {}
        self.actor_optimizers = {}
        self.critic_optimizers = {}
        self.target_actors = {}
        self.target_critics = {}
        
        self._initialize_networks(actor_lr, critic_lr)
    
    def _initialize_networks(self, actor_lr, critic_lr):
        from core.policy_network import PolicyNetwork
        from core.value_network import CentralizedValueNetwork
        
        for i in range(self.num_agents):
            agent_id = f'agent_{i}'
            
            actor = PolicyNetwork(self.state_dim, self.action_dim)
            critic = CentralizedValueNetwork(self.num_agents, self.state_dim)
            
            target_actor = PolicyNetwork(self.state_dim, self.action_dim)
            target_critic = CentralizedValueNetwork(self.num_agents, self.state_dim)
            
            target_actor.load_state_dict(actor.state_dict())
            target_critic.load_state_dict(critic.state_dict())
            
            actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
            critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
            
            self.actors[agent_id] = actor
            self.critics[agent_id] = critic
            self.target_actors[agent_id] = target_actor
            self.target_critics[agent_id] = target_critic
            self.actor_optimizers[agent_id] = actor_optimizer
            self.critic_optimizers[agent_id] = critic_optimizer
    
    def select_actions(self, observations: Dict[str, np.ndarray], noise_scale=0.1, training=True) -> Dict[str, int]:
        actions = {}
        
        for agent_id, observation in observations.items():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            
            action_probs = self.actors[agent_id](obs_tensor)
            
            if training:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                
                if noise_scale > 0:
                    noise = torch.randn_like(action_probs) * noise_scale
                    action_probs = torch.softmax(action_probs + noise, dim=-1)
                    action = torch.distributions.Categorical(action_probs).sample()
            else:
                action = torch.argmax(action_probs, dim=-1)
            
            actions[agent_id] = action.item()
        
        return actions
    
    def update(self, batch_data: Dict[str, Any]):
        observations = batch_data['observations']
        actions = batch_data['actions']
        rewards = batch_data['rewards']
        next_observations = batch_data['next_observations']
        dones = batch_data['dones']
        
        for agent_id in self.actors.keys():
            if agent_id not in observations:
                continue
                
            self._update_critic(agent_id, observations, actions, rewards, next_observations, dones)
            self._update_actor(agent_id, observations)
            self._update_target_networks(agent_id)
    
    def _update_critic(self, agent_id, observations, actions, rewards, next_observations, dones):
        critic = self.critics[agent_id]
        target_critic = self.target_critics[agent_id]
        
        batch_size = len(observations[agent_id])
        
        obs_tensor = self._prepare_centralized_input(observations)
        actions_tensor = self._prepare_actions_tensor(actions)
        next_obs_tensor = self._prepare_centralized_input(next_observations)
        rewards_tensor = torch.FloatTensor(rewards[agent_id])
        dones_tensor = torch.BoolTensor(dones[agent_id])
        
        with torch.no_grad():
            next_actions = []
            for other_agent_id in self.actors.keys():
                if other_agent_id in next_observations:
                    next_obs_other = torch.FloatTensor(next_observations[other_agent_id])
                    next_action_probs = self.target_actors[other_agent_id](next_obs_other)
                    next_actions.append(torch.argmax(next_action_probs, dim=-1).unsqueeze(1))
            
            next_actions_tensor = torch.cat(next_actions, dim=1)
            next_q_value = target_critic(next_obs_tensor)
            target_q_value = rewards_tensor + self.gamma * next_q_value.squeeze() * (~dones_tensor).float()
        
        current_q_value = critic(obs_tensor).squeeze()
        critic_loss = nn.MSELoss()(current_q_value, target_q_value.detach())
        
        self.critic_optimizers[agent_id].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        self.critic_optimizers[agent_id].step()
    
    def _update_actor(self, agent_id, observations):
        actor = self.actors[agent_id]
        
        obs_tensor = torch.FloatTensor(observations[agent_id])
        centralized_obs_tensor = self._prepare_centralized_input(observations)
        
        action_probs = actor(obs_tensor)
        actions = torch.argmax(action_probs, dim=-1)
        
        actions_tensor = self._prepare_actions_tensor({agent_id: actions}, batch_size=len(obs_tensor))
        
        q_value = self.critics[agent_id](centralized_obs_tensor)
        actor_loss = -q_value.mean()
        
        self.actor_optimizers[agent_id].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        self.actor_optimizers[agent_id].step()
    
    def _update_target_networks(self, agent_id):
        for param, target_param in zip(self.actors[agent_id].parameters(), self.target_actors[agent_id].parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critics[agent_id].parameters(), self.target_critics[agent_id].parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _prepare_centralized_input(self, observations: Dict[str, np.ndarray]) -> torch.Tensor:
        batch_size = len(next(iter(observations.values())))
        centralized_obs = []
        
        for i in range(batch_size):
            episode_obs = []
            for agent_id in sorted(observations.keys()):
                episode_obs.extend(observations[agent_id][i])
            centralized_obs.append(episode_obs)
        
        return torch.FloatTensor(centralized_obs)
    
    def _prepare_actions_tensor(self, actions: Dict[str, np.ndarray], batch_size=None) -> torch.Tensor:
        if batch_size is None:
            batch_size = len(next(iter(actions.values())))
        
        actions_tensor = []
        for i in range(batch_size):
            episode_actions = []
            for agent_id in sorted(actions.keys()):
                episode_actions.append(actions[agent_id][i])
            actions_tensor.append(episode_actions)
        
        return torch.LongTensor(actions_tensor)
    
    def save_models(self, filepath: str):
        models = {}
        for agent_id in self.actors.keys():
            models[agent_id] = {
                'actor_state_dict': self.actors[agent_id].state_dict(),
                'critic_state_dict': self.critics[agent_id].state_dict(),
                'target_actor_state_dict': self.target_actors[agent_id].state_dict(),
                'target_critic_state_dict': self.target_critics[agent_id].state_dict()
            }
        torch.save(models, filepath)
    
    def load_models(self, filepath: str):
        models = torch.load(filepath)
        for agent_id, model_data in models.items():
            if agent_id in self.actors:
                self.actors[agent_id].load_state_dict(model_data['actor_state_dict'])
                self.critics[agent_id].load_state_dict(model_data['critic_state_dict'])
                self.target_actors[agent_id].load_state_dict(model_data['target_actor_state_dict'])
                self.target_critics[agent_id].load_state_dict(model_data['target_critic_state_dict'])