import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import PredatorPreyEnv, MultiAgentReplayBuffer
from algorithms import MADDPG, IQL
from utils import TrainingUtils, EvaluationUtils, VisualizationUtils

def competitive_training_example():
    print("=== Competitive Multi-Agent Reinforcement Learning Training ===")
    
    num_predators = 2
    num_prey = 1
    num_agents = num_predators + num_prey
    state_dim = 15
    action_dim = 5
    max_episodes = 1500
    max_steps = 300
    
    env = PredatorPreyEnv(num_predators=num_predators, num_prey=num_prey, state_dim=state_dim, action_dim=action_dim, max_steps=max_steps)
    replay_buffer = MultiAgentReplayBuffer(capacity=15000, num_agents=num_agents)
    
    predator_algorithm = MADDPG(num_predators, state_dim, action_dim)
    prey_algorithm = IQL(num_prey, state_dim, action_dim)
    
    training_history = []
    best_predator_success_rate = 0.0
    
    print(f"Training predators vs prey scenario...")
    print(f"Predators: {num_predators}, Prey: {num_prey}")
    
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
        
        prey_caught = False
        
        for step in range(max_steps):
            predator_observations = {k: v for k, v in observations.items() if int(k.split('_')[1]) < num_predators}
            prey_observations = {k: v for k, v in observations.items() if int(k.split('_')[1]) >= num_predators}
            
            predator_actions = predator_algorithm.select_actions(predator_observations, training=True)
            prey_actions = prey_algorithm.select_actions(prey_observations, training=True)
            
            actions = {**predator_actions, **prey_actions}
            
            log_probs = {}
            values = {}
            for agent_id in actions.keys():
                log_probs[agent_id] = torch.tensor(0.0)
                values[agent_id] = torch.tensor(0.0)
            
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
                prey_caught = any(env.prey_caught)
                break
        
        predator_replay_data = {}
        prey_replay_data = {}
        
        if len(replay_buffer) >= 2000:
            batch_data = replay_buffer.sample(512)
            
            for agent_id in batch_data['observations'].keys():
                agent_idx = int(agent_id.split('_')[1])
                if agent_idx < num_predators:
                    if 'predator' not in predator_replay_data:
                        predator_replay_data = {k: {} for k in batch_data.keys()}
                    for key in batch_data.keys():
                        predator_replay_data[key][agent_id] = batch_data[key][agent_id]
                else:
                    if 'prey' not in prey_replay_data:
                        prey_replay_data = {k: {} for k in batch_data.keys()}
                    for key in batch_data.keys():
                        prey_replay_data[key][agent_id] = batch_data[key][agent_id]
            
            if predator_replay_data and any(predator_replay_data['observations']):
                predator_algorithm.update(predator_replay_data)
            
            if prey_replay_data and any(prey_replay_data['observations']):
                prey_algorithm.update(prey_replay_data)
        
        total_predator_reward = sum(episode_rewards[f'agent_{i}'] for i in range(num_predators))
        total_prey_reward = sum(episode_rewards[f'agent_{i}'] for i in range(num_predators, num_agents))
        
        predator_success_rate = 1.0 if prey_caught else 0.0
        
        episode_history = {
            'episode': episode,
            'total_predator_reward': total_predator_reward,
            'total_prey_reward': total_prey_reward,
            'predator_success_rate': predator_success_rate,
            'prey_caught': prey_caught
        }
        
        training_history.append(episode_history)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, "
                  f"Predator Reward: {total_predator_reward:.2f}, "
                  f"Prey Reward: {total_prey_reward:.2f}, "
                  f"Prey Caught: {prey_caught}")
        
        if predator_success_rate > best_predator_success_rate:
            best_predator_success_rate = predator_success_rate
            predator_algorithm.save_models('best_predator_model.pth')
            prey_algorithm.save_models('best_prey_model.pth')
    
    visualizer = VisualizationUtils()
    
    predator_rewards = [ep['total_predator_reward'] for ep in training_history]
    prey_rewards = [ep['total_prey_reward'] for ep in training_history]
    success_rates = [ep['predator_success_rate'] for ep in training_history]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(predator_rewards, label='Predators', color='red')
    axes[0].set_title('Predator Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(prey_rewards, label='Prey', color='blue')
    axes[1].set_title('Prey Rewards')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Total Reward')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(success_rates, label='Success Rate', color='green')
    axes[2].set_title('Predator Success Rate')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Success Rate')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTraining completed. Best predator success rate: {best_predator_success_rate:.2f}")
    
    return predator_algorithm, prey_algorithm, training_history

if __name__ == "__main__":
    predator_algo, prey_algo, history = competitive_training_example()