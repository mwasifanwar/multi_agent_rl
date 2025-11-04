import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class QMIX:
    def __init__(self, num_agents, state_dim, action_dim, mixing_hidden=32, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        from core.value_network import QMIXNetwork
        self.qmix_net = QMIXNetwork(num_agents, state_dim, action_dim, hidden_dim=128, mixing_hidden=mixing_hidden)
        self.target_qmix_net = QMIXNetwork(num_agents, state_dim, action_dim, hidden_dim=128, mixing_hidden=mixing_hidden)
        
        self.target_qmix_net.load_state_dict(self.qmix_net.state_dict())
        
        self.optimizer = optim.Adam(self.qmix_net.parameters(), lr=learning_rate)
        
        self.global_state_dim = state_dim * num_agents
    
    def select_actions(self, observations: Dict[str, np.ndarray], training=True) -> Dict[str, int]:
        actions = {}
        
        if training and np.random.random() < self.epsilon:
            for agent_id in observations.keys():
                actions[agent_id] = np.random.randint(self.action_dim)
        else:
            batch_observations = self._prepare_batch_observations(observations)
            global_state = self._get_global_state(observations)
            
            with torch.no_grad():
                _, agent_q_values = self.qmix_net(batch_observations, global_state)
                
                for i, agent_id in enumerate(sorted(observations.keys())):
                    actions[agent_id] = torch.argmax(agent_q_values[0, i]).item()
        
        return actions
    
    def update(self, batch_data: Dict[str, Any]):
        observations = batch_data['observations']
        actions = batch_data['actions']
        rewards = batch_data['rewards']
        next_observations = batch_data['next_observations']
        dones = batch_data['dones']
        
        batch_size = len(observations[list(observations.keys())[0]])
        
        batch_obs = self._prepare_batch_observations(observations, batch_size)
        batch_next_obs = self._prepare_batch_observations(next_observations, batch_size)
        global_state = self._get_global_state_batch(observations, batch_size)
        next_global_state = self._get_global_state_batch(next_observations, batch_size)
        
        actions_tensor = self._prepare_actions_tensor(actions, batch_size)
        rewards_tensor = torch.FloatTensor(rewards[list(rewards.keys())[0]])
        dones_tensor = torch.BoolTensor(dones[list(dones.keys())[0]])
        
        current_q_total, current_agent_qs = self.qmix_net(batch_obs, global_state)
        
        with torch.no_grad():
            _, next_agent_qs = self.target_qmix_net(batch_next_obs, next_global_state)
            next_actions = torch.argmax(next_agent_qs, dim=-1)
            next_max_qs = next_agent_qs.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)
            
            next_q_total, _ = self.target_qmix_net(batch_next_obs, next_global_state)
            target_q_total = rewards_tensor + self.gamma * next_q_total * (~dones_tensor).float()
        
        current_agent_qs = current_agent_qs.gather(2, actions_tensor.unsqueeze(-1)).squeeze(-1)
        
        loss = nn.MSELoss()(current_q_total, target_q_total.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qmix_net.parameters(), 10.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _prepare_batch_observations(self, observations: Dict[str, np.ndarray], batch_size=1) -> torch.Tensor:
        batch_obs = []
        for i in range(batch_size):
            episode_obs = []
            for agent_id in sorted(observations.keys()):
                episode_obs.extend(observations[agent_id][i])
            batch_obs.append(episode_obs)
        return torch.FloatTensor(batch_obs)
    
    def _get_global_state(self, observations: Dict[str, np.ndarray]) -> torch.Tensor:
        global_state = []
        for agent_id in sorted(observations.keys()):
            global_state.extend(observations[agent_id])
        return torch.FloatTensor(global_state).unsqueeze(0)
    
    def _get_global_state_batch(self, observations: Dict[str, np.ndarray], batch_size: int) -> torch.Tensor:
        global_states = []
        for i in range(batch_size):
            episode_global_state = []
            for agent_id in sorted(observations.keys()):
                episode_global_state.extend(observations[agent_id][i])
            global_states.append(episode_global_state)
        return torch.FloatTensor(global_states)
    
    def _prepare_actions_tensor(self, actions: Dict[str, np.ndarray], batch_size: int) -> torch.Tensor:
        actions_tensor = []
        for i in range(batch_size):
            episode_actions = []
            for agent_id in sorted(actions.keys()):
                episode_actions.append(actions[agent_id][i])
            actions_tensor.append(episode_actions)
        return torch.LongTensor(actions_tensor).unsqueeze(-1)
    
    def update_target_network(self):
        self.target_qmix_net.load_state_dict(self.qmix_net.state_dict())
    
    def save_model(self, filepath: str):
        torch.save({
            'qmix_net_state_dict': self.qmix_net.state_dict(),
            'target_qmix_net_state_dict': self.target_qmix_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath)
        self.qmix_net.load_state_dict(checkpoint['qmix_net_state_dict'])
        self.target_qmix_net.load_state_dict(checkpoint['target_qmix_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)