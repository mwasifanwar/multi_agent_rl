import numpy as np
import random
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class MultiAgentReplayBuffer:
    def __init__(self, capacity=10000, num_agents=3):
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffer = []
        self.position = 0
    
    def push(self, 
             observations: Dict[str, np.ndarray],
             actions: Dict[str, int],
             rewards: Dict[str, float],
             next_observations: Dict[str, np.ndarray],
             dones: Dict[str, bool],
             log_probs: Dict[str, Any],
             values: Dict[str, Any]):
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        experience = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_observations,
            'dones': dones,
            'log_probs': log_probs,
            'values': values
        }
        
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        batch_data = {
            'observations': {},
            'actions': {},
            'rewards': {},
            'next_observations': {},
            'dones': {},
            'log_probs': {},
            'values': {}
        }
        
        for agent_id in [f'agent_{i}' for i in range(self.num_agents)]:
            batch_data['observations'][agent_id] = []
            batch_data['actions'][agent_id] = []
            batch_data['rewards'][agent_id] = []
            batch_data['next_observations'][agent_id] = []
            batch_data['dones'][agent_id] = []
            batch_data['log_probs'][agent_id] = []
            batch_data['values'][agent_id] = []
        
        for experience in batch:
            for agent_id in experience['observations'].keys():
                batch_data['observations'][agent_id].append(experience['observations'][agent_id])
                batch_data['actions'][agent_id].append(experience['actions'][agent_id])
                batch_data['rewards'][agent_id].append(experience['rewards'][agent_id])
                batch_data['next_observations'][agent_id].append(experience['next_observations'][agent_id])
                batch_data['dones'][agent_id].append(experience['dones'][agent_id])
                batch_data['log_probs'][agent_id].append(experience['log_probs'][agent_id])
                batch_data['values'][agent_id].append(experience['values'][agent_id])
        
        for agent_id in batch_data['observations'].keys():
            if batch_data['observations'][agent_id]:
                batch_data['observations'][agent_id] = np.array(batch_data['observations'][agent_id])
                batch_data['actions'][agent_id] = np.array(batch_data['actions'][agent_id])
                batch_data['rewards'][agent_id] = np.array(batch_data['rewards'][agent_id])
                batch_data['next_observations'][agent_id] = np.array(batch_data['next_observations'][agent_id])
                batch_data['dones'][agent_id] = np.array(batch_data['dones'][agent_id])
        
        return batch_data
    
    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer(MultiAgentReplayBuffer):
    def __init__(self, capacity=10000, num_agents=3, alpha=0.6, beta=0.4):
        super().__init__(capacity, num_agents)
        self.alpha = alpha
        self.beta = beta
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0
    
    def push(self, observations, actions, rewards, next_observations, dones, log_probs, values, priority=None):
        if priority is None:
            priority = self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities = np.append(self.priorities, 0)
        
        experience = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_observations,
            'dones': dones,
            'log_probs': log_probs,
            'values': values
        }
        
        self.buffer[self.position] = experience
        self.priorities[self.position] = priority
        self.max_priority = max(self.max_priority, priority)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        if len(self.buffer) == 0:
            return {}
        
        probabilities = self.priorities[:len(self.buffer)] ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        batch = [self.buffer[idx] for idx in indices]
        
        batch_data = {
            'observations': {},
            'actions': {},
            'rewards': {},
            'next_observations': {},
            'dones': {},
            'log_probs': {},
            'values': {},
            'weights': weights,
            'indices': indices
        }
        
        for agent_id in [f'agent_{i}' for i in range(self.num_agents)]:
            batch_data['observations'][agent_id] = []
            batch_data['actions'][agent_id] = []
            batch_data['rewards'][agent_id] = []
            batch_data['next_observations'][agent_id] = []
            batch_data['dones'][agent_id] = []
            batch_data['log_probs'][agent_id] = []
            batch_data['values'][agent_id] = []
        
        for experience in batch:
            for agent_id in experience['observations'].keys():
                batch_data['observations'][agent_id].append(experience['observations'][agent_id])
                batch_data['actions'][agent_id].append(experience['actions'][agent_id])
                batch_data['rewards'][agent_id].append(experience['rewards'][agent_id])
                batch_data['next_observations'][agent_id].append(experience['next_observations'][agent_id])
                batch_data['dones'][agent_id].append(experience['dones'][agent_id])
                batch_data['log_probs'][agent_id].append(experience['log_probs'][agent_id])
                batch_data['values'][agent_id].append(experience['values'][agent_id])
        
        for agent_id in batch_data['observations'].keys():
            if batch_data['observations'][agent_id]:
                batch_data['observations'][agent_id] = np.array(batch_data['observations'][agent_id])
                batch_data['actions'][agent_id] = np.array(batch_data['actions'][agent_id])
                batch_data['rewards'][agent_id] = np.array(batch_data['rewards'][agent_id])
                batch_data['next_observations'][agent_id] = np.array(batch_data['next_observations'][agent_id])
                batch_data['dones'][agent_id] = np.array(batch_data['dones'][agent_id])
        
        return batch_data
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)