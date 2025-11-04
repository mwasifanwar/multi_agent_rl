import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class VisualizationUtils:
    def __init__(self, style='seaborn'):
        self.style = style
        plt.style.use(style)
    
    def plot_learning_curves(self, training_history: Dict[str, List[float]], window_size=10):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = range(len(training_history.get('episode_rewards', [])))
        
        if 'episode_rewards' in training_history:
            rewards = training_history['episode_rewards']
            smoothed_rewards = self._smooth_data(rewards, window_size)
            
            axes[0, 0].plot(episodes, rewards, alpha=0.3, color='blue')
            axes[0, 0].plot(episodes[window_size-1:], smoothed_rewards, color='blue', linewidth=2)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].grid(True, alpha=0.3)
        
        if 'average_returns' in training_history:
            returns = training_history['average_returns']
            smoothed_returns = self._smooth_data(returns, window_size)
            
            axes[0, 1].plot(episodes, returns, alpha=0.3, color='green')
            axes[0, 1].plot(episodes[window_size-1:], smoothed_returns, color='green', linewidth=2)
            axes[0, 1].set_title('Average Returns')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Return')
            axes[0, 1].grid(True, alpha=0.3)
        
        if 'cooperation_scores' in training_history:
            cooperation = training_history['cooperation_scores']
            smoothed_cooperation = self._smooth_data(cooperation, window_size)
            
            axes[1, 0].plot(episodes, cooperation, alpha=0.3, color='red')
            axes[1, 0].plot(episodes[window_size-1:], smoothed_cooperation, color='red', linewidth=2)
            axes[1, 0].set_title('Cooperation Scores')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Cooperation Score')
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'exploration_rates' in training_history:
            exploration = training_history['exploration_rates']
            
            axes[1, 1].plot(episodes, exploration, color='orange', linewidth=2)
            axes[1, 1].set_title('Exploration Rates')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Exploration Rate')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_agent_performance(self, agent_performance: Dict[str, List[float]]):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        episodes = range(len(next(iter(agent_performance.values()))))
        
        for agent_id, performance in agent_performance.items():
            axes[0].plot(episodes, performance, label=agent_id, linewidth=2)
        
        axes[0].set_title('Individual Agent Performance')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Performance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        final_performance = {agent_id: perf[-1] for agent_id, perf in agent_performance.items()}
        agents = list(final_performance.keys())
        performances = list(final_performance.values())
        
        axes[1].bar(agents, performances, color='skyblue', alpha=0.7)
        axes[1].set_title('Final Agent Performance')
        axes[1].set_xlabel('Agent')
        axes[1].set_ylabel('Final Performance')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_emergence_analysis(self, emergence_data: Dict[str, Any]):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        if 'strategy_complexity' in emergence_data:
            complexity = emergence_data['strategy_complexity']
            axes[0, 0].plot(range(len(complexity)), complexity, linewidth=2, color='purple')
            axes[0, 0].set_title('Strategy Complexity Over Time')
            axes[0, 0].set_xlabel('Training Phase')
            axes[0, 0].set_ylabel('Complexity')
            axes[0, 0].grid(True, alpha=0.3)
        
        if 'coordination_efficiency' in emergence_data:
            efficiency = emergence_data['coordination_efficiency']
            axes[0, 1].plot(range(len(efficiency)), efficiency, linewidth=2, color='teal')
            axes[0, 1].set_title('Coordination Efficiency')
            axes[0, 1].set_xlabel('Training Phase')
            axes[0, 1].set_ylabel('Efficiency')
            axes[0, 1].grid(True, alpha=0.3)
        
        if 'role_specialization' in emergence_data:
            specialization = emergence_data['role_specialization']
            axes[1, 0].plot(range(len(specialization)), specialization, linewidth=2, color='orange')
            axes[1, 0].set_title('Role Specialization')
            axes[1, 0].set_xlabel('Training Phase')
            axes[1, 0].set_ylabel('Specialization')
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'communication_patterns' in emergence_data:
            communication = emergence_data['communication_patterns']
            axes[1, 1].plot(range(len(communication)), communication, linewidth=2, color='red')
            axes[1, 1].set_title('Communication Pattern Development')
            axes[1, 1].set_xlabel('Training Phase')
            axes[1, 1].set_ylabel('Pattern Strength')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_action_distributions(self, action_data: Dict[str, List[int]]):
        fig, axes = plt.subplots(1, len(action_data), figsize=(5*len(action_data), 5))
        
        if len(action_data) == 1:
            axes = [axes]
        
        for idx, (agent_id, actions) in enumerate(action_data.items()):
            action_counts = np.bincount(actions, minlength=10)
            action_probs = action_counts / len(actions) if actions else np.zeros(10)
            
            axes[idx].bar(range(len(action_probs)), action_probs, alpha=0.7, color='lightblue')
            axes[idx].set_title(f'Action Distribution - {agent_id}')
            axes[idx].set_xlabel('Action')
            axes[idx].set_ylabel('Probability')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _smooth_data(self, data: List[float], window_size: int) -> List[float]:
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            smoothed.append(np.mean(window))
        
        return smoothed
    
    def create_animation_frames(self, episode_data: List[Dict[str, Any]], env):
        frames = []
        
        for step_data in episode_data:
            env.render()
            frame = plt.gcf()
            frames.append(frame)
            plt.pause(0.01)
        
        return frames