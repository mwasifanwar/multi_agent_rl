import numpy as np
import torch
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class TrainingUtils:
    def __init__(self):
        pass
    
    @staticmethod
    def compute_gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray, dones: np.ndarray, gamma: float = 0.99, gae_lambda: float = 0.95) -> np.ndarray:
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
        
        return advantages
    
    @staticmethod
    def normalize_advantages(advantages: np.ndarray) -> np.ndarray:
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages
    
    @staticmethod
    def compute_returns(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
        returns = np.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        return returns
    
    @staticmethod
    def create_mini_batches(experience_data: Dict[str, Any], mini_batch_size: int) -> List[Dict[str, Any]]:
        batch_size = len(experience_data['observations'][list(experience_data['observations'].keys())[0]])
        mini_batches = []
        
        for start_idx in range(0, batch_size, mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, batch_size)
            mini_batch = {}
            
            for key in experience_data.keys():
                if key in ['observations', 'actions', 'rewards', 'next_observations', 'dones']:
                    mini_batch[key] = {}
                    for agent_id in experience_data[key].keys():
                        mini_batch[key][agent_id] = experience_data[key][agent_id][start_idx:end_idx]
                elif key in ['log_probs', 'values']:
                    mini_batch[key] = {}
                    for agent_id in experience_data[key].keys():
                        mini_batch[key][agent_id] = experience_data[key][agent_id][start_idx:end_idx]
                else:
                    mini_batch[key] = experience_data[key][start_idx:end_idx]
            
            mini_batches.append(mini_batch)
        
        return mini_batches
    
    @staticmethod
    def update_learning_rate(optimizer, new_lr: float):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    
    @staticmethod
    def compute_gradient_norm(model) -> float:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    @staticmethod
    def clip_gradients(model, max_norm: float = 1.0):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    @staticmethod
    def compute_parameter_norm(model) -> float:
        total_norm = 0.0
        for p in model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

class CurriculumLearning:
    def __init__(self, difficulty_levels: List[float], progression_threshold: float = 0.8):
        self.difficulty_levels = difficulty_levels
        self.progression_threshold = progression_threshold
        self.current_level = 0
        self.success_rate = 0.0
        self.episode_count = 0
        self.success_count = 0
    
    def update_success_rate(self, success: bool):
        self.episode_count += 1
        if success:
            self.success_count += 1
        
        if self.episode_count >= 10:
            self.success_rate = self.success_count / self.episode_count
            self.episode_count = 0
            self.success_count = 0
    
    def should_progress(self) -> bool:
        return self.success_rate >= self.progression_threshold and self.current_level < len(self.difficulty_levels) - 1
    
    def should_regress(self) -> bool:
        return self.success_rate < self.progression_threshold / 2 and self.current_level > 0
    
    def get_current_difficulty(self) -> float:
        return self.difficulty_levels[self.current_level]
    
    def progress_level(self):
        if self.current_level < len(self.difficulty_levels) - 1:
            self.current_level += 1
            return True
        return False
    
    def regress_level(self):
        if self.current_level > 0:
            self.current_level -= 1
            return True
        return False