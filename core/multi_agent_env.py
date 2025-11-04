import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class MultiAgentEnvironment:
    def __init__(self, num_agents=3, state_dim=10, action_dim=4, max_steps=1000):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.current_step = 0
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
        self.action_space = spaces.Discrete(action_dim)
        
        self.agent_positions = None
        self.target_positions = None
        self.obstacles = None
        
        self.reset()
    
    def reset(self) -> Dict[str, np.ndarray]:
        self.current_step = 0
        
        self.agent_positions = np.random.uniform(-1, 1, (self.num_agents, 2))
        self.target_positions = np.random.uniform(-1, 1, (self.num_agents, 2))
        
        num_obstacles = 5
        self.obstacles = np.random.uniform(-1, 1, (num_obstacles, 2))
        
        observations = {}
        for i in range(self.num_agents):
            observations[f'agent_{i}'] = self._get_agent_observation(i)
            
        return observations
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        self.current_step += 1
        
        for agent_id, action in actions.items():
            agent_idx = int(agent_id.split('_')[1])
            self._update_agent_position(agent_idx, action)
        
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        for i in range(self.num_agents):
            observations[f'agent_{i}'] = self._get_agent_observation(i)
            rewards[f'agent_{i}'] = self._compute_agent_reward(i)
            dones[f'agent_{i}'] = self.current_step >= self.max_steps
            infos[f'agent_{i}'] = {}
        
        global_done = all(dones.values())
        if global_done:
            for i in range(self.num_agents):
                dones[f'agent_{i}'] = True
                
        return observations, rewards, dones, infos
    
    def _get_agent_observation(self, agent_idx: int) -> np.ndarray:
        obs = np.zeros(self.state_dim)
        
        obs[0:2] = self.agent_positions[agent_idx]
        obs[2:4] = self.target_positions[agent_idx]
        
        other_agent_idx = 0
        for i in range(self.num_agents):
            if i != agent_idx:
                start_idx = 4 + other_agent_idx * 2
                obs[start_idx:start_idx+2] = self.agent_positions[i]
                other_agent_idx += 1
        
        obstacle_start = 4 + (self.num_agents - 1) * 2
        for i in range(min(5, len(self.obstacles))):
            start_idx = obstacle_start + i * 2
            obs[start_idx:start_idx+2] = self.obstacles[i]
        
        return obs
    
    def _update_agent_position(self, agent_idx: int, action: int):
        move_vectors = {
            0: np.array([0.1, 0.0]),
            1: np.array([-0.1, 0.0]),
            2: np.array([0.0, 0.1]),
            3: np.array([0.0, -0.1])
        }
        
        if action in move_vectors:
            new_position = self.agent_positions[agent_idx] + move_vectors[action]
            
            if not self._check_collision(new_position):
                self.agent_positions[agent_idx] = new_position
    
    def _check_collision(self, position: np.ndarray) -> bool:
        for obstacle in self.obstacles:
            if np.linalg.norm(position - obstacle) < 0.2:
                return True
        return False
    
    def _compute_agent_reward(self, agent_idx: int) -> float:
        reward = 0.0
        
        target_dist = np.linalg.norm(self.agent_positions[agent_idx] - self.target_positions[agent_idx])
        reward += -target_dist
        
        for i, obstacle in enumerate(self.obstacles):
            obstacle_dist = np.linalg.norm(self.agent_positions[agent_idx] - obstacle)
            if obstacle_dist < 0.3:
                reward -= 1.0 / (obstacle_dist + 1e-6)
        
        if target_dist < 0.1:
            reward += 10.0
            
        return reward
    
    def render(self, mode='human'):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        
        for i, pos in enumerate(self.agent_positions):
            circle = patches.Circle(pos, 0.05, fill=True, color='blue', alpha=0.7)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1] + 0.08, f'A{i}', ha='center', va='center', fontsize=8)
        
        for i, pos in enumerate(self.target_positions):
            circle = patches.Circle(pos, 0.05, fill=True, color='green', alpha=0.7)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1] + 0.08, f'T{i}', ha='center', va='center', fontsize=8)
        
        for i, pos in enumerate(self.obstacles):
            circle = patches.Circle(pos, 0.1, fill=True, color='red', alpha=0.5)
            ax.add_patch(circle)
        
        plt.title(f'Multi-Agent Environment - Step {self.current_step}')
        plt.grid(True, alpha=0.3)
        
        if mode == 'human':
            plt.pause(0.01)
        else:
            plt.show()

class CooperativeNavigationEnv(MultiAgentEnvironment):
    def __init__(self, num_agents=3, num_targets=3, state_dim=20, action_dim=5):
        super().__init__(num_agents, state_dim, action_dim)
        self.num_targets = num_targets
        self.target_assignments = None
        
    def reset(self):
        super().reset()
        
        self.target_positions = np.random.uniform(-1, 1, (self.num_targets, 2))
        self.target_assignments = np.random.choice(self.num_targets, self.num_agents, replace=False)
        
        observations = {}
        for i in range(self.num_agents):
            observations[f'agent_{i}'] = self._get_agent_observation(i)
            
        return observations
    
    def _get_agent_observation(self, agent_idx: int) -> np.ndarray:
        obs = super()._get_agent_observation(agent_idx)
        
        assigned_target = self.target_assignments[agent_idx]
        obs[4:6] = self.target_positions[assigned_target]
        
        return obs
    
    def _compute_agent_reward(self, agent_idx: int) -> float:
        reward = 0.0
        
        assigned_target = self.target_assignments[agent_idx]
        target_dist = np.linalg.norm(self.agent_positions[agent_idx] - self.target_positions[assigned_target])
        reward += -target_dist
        
        for other_idx in range(self.num_agents):
            if other_idx != agent_idx:
                other_dist = np.linalg.norm(self.agent_positions[agent_idx] - self.agent_positions[other_idx])
                if other_dist < 0.3:
                    reward -= 0.5
        
        if target_dist < 0.1:
            reward += 15.0
            
        return reward

class PredatorPreyEnv(MultiAgentEnvironment):
    def __init__(self, num_predators=2, num_prey=1, state_dim=15, action_dim=5):
        super().__init__(num_predators + num_prey, state_dim, action_dim)
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.prey_caught = [False] * num_prey
        
    def reset(self):
        self.current_step = 0
        self.prey_caught = [False] * self.num_prey
        
        self.agent_positions = np.random.uniform(-1, 1, (self.num_agents, 2))
        
        observations = {}
        for i in range(self.num_agents):
            observations[f'agent_{i}'] = self._get_agent_observation(i)
            
        return observations
    
    def _get_agent_observation(self, agent_idx: int) -> np.ndarray:
        obs = np.zeros(self.state_dim)
        
        obs[0:2] = self.agent_positions[agent_idx]
        
        prey_idx = 0
        predator_idx = 0
        
        for i in range(self.num_agents):
            if i < self.num_predators:
                if i != agent_idx:
                    start_idx = 2 + predator_idx * 2
                    obs[start_idx:start_idx+2] = self.agent_positions[i]
                    predator_idx += 1
            else:
                prey_global_idx = i - self.num_predators
                if not self.prey_caught[prey_global_idx]:
                    start_idx = 2 + (self.num_predators - 1) * 2 + prey_idx * 2
                    obs[start_idx:start_idx+2] = self.agent_positions[i]
                    prey_idx += 1
        
        return obs
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        self.current_step += 1
        
        for agent_id, action in actions.items():
            agent_idx = int(agent_id.split('_')[1])
            self._update_agent_position(agent_idx, action)
        
        self._check_prey_capture()
        
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        for i in range(self.num_agents):
            observations[f'agent_{i}'] = self._get_agent_observation(i)
            rewards[f'agent_{i}'] = self._compute_agent_reward(i)
            dones[f'agent_{i}'] = self.current_step >= self.max_steps or all(self.prey_caught)
            infos[f'agent_{i}'] = {}
        
        return observations, rewards, dones, infos
    
    def _check_prey_capture(self):
        for prey_idx in range(self.num_prey):
            if not self.prey_caught[prey_idx]:
                prey_agent_idx = self.num_predators + prey_idx
                prey_pos = self.agent_positions[prey_agent_idx]
                
                predator_count = 0
                for predator_idx in range(self.num_predators):
                    predator_pos = self.agent_positions[predator_idx]
                    if np.linalg.norm(prey_pos - predator_pos) < 0.2:
                        predator_count += 1
                
                if predator_count >= 2:
                    self.prey_caught[prey_idx] = True
    
    def _compute_agent_reward(self, agent_idx: int) -> float:
        reward = 0.0
        
        if agent_idx < self.num_predators:
            for prey_idx in range(self.num_prey):
                if not self.prey_caught[prey_idx]:
                    prey_agent_idx = self.num_predators + prey_idx
                    prey_dist = np.linalg.norm(self.agent_positions[agent_idx] - self.agent_positions[prey_agent_idx])
                    reward += -prey_dist * 0.1
                else:
                    reward += 10.0
        else:
            prey_idx = agent_idx - self.num_predators
            if self.prey_caught[prey_idx]:
                reward -= 10.0
            else:
                min_predator_dist = float('inf')
                for predator_idx in range(self.num_predators):
                    predator_dist = np.linalg.norm(self.agent_positions[agent_idx] - self.agent_positions[predator_idx])
                    min_predator_dist = min(min_predator_dist, predator_dist)
                
                reward += min_predator_dist * 0.1
                
        return reward