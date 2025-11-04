import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import MultiAgentEnvironment, CooperativeNavigationEnv, MultiAgentReplayBuffer
from algorithms import MADDPG, QMIX, MAPPO
from utils import TrainingUtils, EvaluationUtils, VisualizationUtils

def cooperative_training_example():
    print("=== Cooperative Multi-Agent Reinforcement Learning Training ===")
    
    num_agents = 3
    state_dim = 20
    action_dim = 5
    max_episodes = 1000
    max_steps = 200
    
    env = CooperativeNavigationEnv(num_agents=num_agents, state_dim=state_dim, action_dim=action_dim, max_steps=max_steps)
    replay_buffer = MultiAgentReplayBuffer(capacity=10000, num_agents=num_agents)
    
    algorithms = {
        'MADDPG': MADDPG(num_agents, state_dim, action_dim),
        'QMIX': QMIX(num_agents, state_dim, action_dim),
        'MAPPO': MAPPO(num_agents, state_dim, action_dim)
    }
    
    selected_algorithm = 'MADDPG'
    algorithm = algorithms[selected_algorithm]
    
    training_history = []
    best_mean_reward = -float('inf')
    
    print(f"Training {selected_algorithm} on cooperative navigation task...")
    print(f"Agents: {num_agents}, State dim: {state_dim}, Action dim: {action_dim}")
    
    for episode in range(max_episodes):
        observations = env.reset()
        episode_rewards = {f'agent_{i}': 0 for i in range(num_agents)}
        episode_data = {
            'observations': {agent_id: [] for agent_id in observations.keys()},
            'actions': {agent_id: [] for agent_id in observations.keys()},
            'rewards': {agent_id: [] for agent_id in observations.keys()},
            'next_observations': {agent_id: [] for agent_id in observations.keys()},
            'dones': {agent_id: [] for agent_id in observations.keys()},
            'log_probs': {agent_id: [] for agent_id in observations.keys()},
            'values': {agent_id: [] for agent_id in observations.keys()}
        }
        
        for step in range(max_steps):
            if selected_algorithm in ['MADDPG', 'IQL']:
                actions = algorithm.select_actions(observations, training=True)
                log_probs = {agent_id: torch.tensor(0.0) for agent_id in actions.keys()}
                values = {agent_id: torch.tensor(0.0) for agent_id in actions.keys()}
            else:
                actions, log_probs, values = algorithm.select_actions(observations, training=True)
            
            next_observations, rewards, dones, infos = env.step(actions)
            
            for agent_id in observations.keys():
                episode_data['observations'][agent_id].append(observations[agent_id])
                episode_data['actions'][agent_id].append(actions[agent_id])
                episode_data['rewards'][agent_id].append(rewards[agent_id])
                episode_data['next_observations'][agent_id].append(next_observations[agent_id])
                episode_data['dones'][agent_id].append(dones[agent_id])
                episode_data['log_probs'][agent_id].append(log_probs[agent_id])
                episode_data['values'][agent_id].append(values[agent_id])
                
                episode_rewards[agent_id] += rewards[agent_id]
            
            replay_buffer.push(observations, actions, rewards, next_observations, dones, log_probs, values)
            
            observations = next_observations
            
            if all(dones.values()):
                break
        
        if len(replay_buffer) >= 1000:
            batch_data = replay_buffer.sample(256)
            algorithm.update(batch_data)
            
            if selected_algorithm == 'QMIX':
                algorithm.update_target_network()
        
        total_episode_reward = sum(episode_rewards.values())
        mean_agent_reward = total_episode_reward / num_agents
        
        cooperation_metrics = EvaluationUtils.compute_cooperation_metrics(episode_data)
        
        episode_history = {
            'episode': episode,
            'total_reward': total_episode_reward,
            'mean_agent_reward': mean_agent_reward,
            'cooperation_score': cooperation_metrics.get('cooperation_score', 0),
            'exploration_rate': getattr(algorithm, 'epsilon', 0) if hasattr(algorithm, 'epsilon') else 0
        }
        
        training_history.append(episode_history)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_episode_reward:.2f}, "
                  f"Mean Agent Reward: {mean_agent_reward:.2f}, "
                  f"Cooperation: {cooperation_metrics.get('cooperation_score', 0):.3f}")
        
        if mean_agent_reward > best_mean_reward:
            best_mean_reward = mean_agent_reward
            algorithm.save_models(f'best_{selected_algorithm}_model.pth')
    
    visualizer = VisualizationUtils()
    learning_curves = EvaluationUtils.compute_learning_curves(training_history)
    fig = visualizer.plot_learning_curves(learning_curves)
    
    print(f"\nTraining completed. Best mean reward: {best_mean_reward:.2f}")
    
    return algorithm, training_history

if __name__ == "__main__":
    algorithm, history = cooperative_training_example()