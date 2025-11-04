import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import MultiAgentEnvironment, MultiAgentReplayBuffer
from algorithms import MAPPO, QMIX
from utils import TrainingUtils, EvaluationUtils, VisualizationUtils, CurriculumLearning

def mixed_training_example():
    print("=== Mixed Cooperative-Competitive Multi-Agent Training ===")
    
    num_agents = 4
    state_dim = 25
    action_dim = 6
    max_episodes = 2000
    max_steps = 250
    
    env = MultiAgentEnvironment(num_agents=num_agents, state_dim=state_dim, action_dim=action_dim, max_steps=max_steps)
    replay_buffer = MultiAgentReplayBuffer(capacity=20000, num_agents=num_agents)
    
    team_a_agents = ['agent_0', 'agent_1']
    team_b_agents = ['agent_2', 'agent_3']
    
    team_a_algorithm = MAPPO(len(team_a_agents), state_dim, action_dim)
    team_b_algorithm = QMIX(len(team_b_agents), state_dim, action_dim)
    
    curriculum = CurriculumLearning(difficulty_levels=[0.1, 0.3, 0.5, 0.7, 1.0])
    
    training_history = []
    best_combined_performance = -float('inf')
    
    print("Training mixed cooperative-competitive scenario...")
    print(f"Team A (MAPPO): {team_a_agents}")
    print(f"Team B (QMIX): {team_b_agents}")
    
    for episode in range(max_episodes):
        current_difficulty = curriculum.get_current_difficulty()
        
        observations = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in observations.keys()}
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
            team_a_obs = {agent_id: observations[agent_id] for agent_id in team_a_agents}
            team_b_obs = {agent_id: observations[agent_id] for agent_id in team_b_agents}
            
            team_a_actions, team_a_log_probs, team_a_values = team_a_algorithm.select_actions(team_a_obs, training=True)
            team_b_actions = team_b_algorithm.select_actions(team_b_obs, training=True)
            
            actions = {**team_a_actions, **team_b_actions}
            log_probs = {**team_a_log_probs, **{agent_id: torch.tensor(0.0) for agent_id in team_b_agents}}
            values = {**team_a_values, **{agent_id: torch.tensor(0.0) for agent_id in team_b_agents}}
            
            next_observations, rewards, dones, infos = env.step(actions)
            
            scaled_rewards = {}
            for agent_id, reward in rewards.items():
                scaled_rewards[agent_id] = reward * current_difficulty
            
            for agent_id in observations.keys():
                episode_data['observations'][agent_id].append(observations[agent_id])
                episode_data['actions'][agent_id].append(actions[agent_id])
                episode_data['rewards'][agent_id].append(scaled_rewards[agent_id])
                episode_data['next_observations'][agent_id].append(next_observations[agent_id])
                episode_data['dones'][agent_id].append(dones[agent_id])
                episode_data['log_probs'][agent_id].append(log_probs[agent_id])
                episode_data['values'][agent_id].append(values[agent_id])
                
                episode_rewards[agent_id] += scaled_rewards[agent_id]
            
            replay_buffer.push(observations, actions, scaled_rewards, next_observations, dones, log_probs, values)
            
            observations = next_observations
            
            if all(dones.values()):
                break
        
        team_a_replay_data = {}
        team_b_replay_data = {}
        
        if len(replay_buffer) >= 3000:
            batch_data = replay_buffer.sample(512)
            
            for agent_id in batch_data['observations'].keys():
                if agent_id in team_a_agents:
                    if 'team_a' not in team_a_replay_data:
                        team_a_replay_data = {k: {} for k in batch_data.keys()}
                    for key in batch_data.keys():
                        team_a_replay_data[key][agent_id] = batch_data[key][agent_id]
                else:
                    if 'team_b' not in team_b_replay_data:
                        team_b_replay_data = {k: {} for k in batch_data.keys()}
                    for key in batch_data.keys():
                        team_b_replay_data[key][agent_id] = batch_data[key][agent_id]
            
            if team_a_replay_data and any(team_a_replay_data['observations']):
                team_a_algorithm.update(team_a_replay_data)
            
            if team_b_replay_data and any(team_b_replay_data['observations']):
                team_b_algorithm.update(team_b_replay_data)
                team_b_algorithm.update_target_network()
        
        team_a_total_reward = sum(episode_rewards[agent_id] for agent_id in team_a_agents)
        team_b_total_reward = sum(episode_rewards[agent_id] for agent_id in team_b_agents)
        combined_performance = team_a_total_reward + team_b_total_reward
        
        cooperation_metrics = EvaluationUtils.compute_cooperation_metrics(episode_data)
        emergence_analysis = EvaluationUtils.analyze_emergent_behavior(episode_data)
        
        curriculum.update_success_rate(combined_performance > 0)
        
        if curriculum.should_progress():
            curriculum.progress_level()
            print(f"Curriculum: Progressed to level {curriculum.current_level}")
        elif curriculum.should_regress():
            curriculum.regress_level()
            print(f"Curriculum: Regressed to level {curriculum.current_level}")
        
        episode_history = {
            'episode': episode,
            'team_a_reward': team_a_total_reward,
            'team_b_reward': team_b_total_reward,
            'combined_performance': combined_performance,
            'difficulty_level': curriculum.current_level,
            'cooperation_score': cooperation_metrics.get('cooperation_score', 0),
            'emergence_metrics': emergence_analysis
        }
        
        training_history.append(episode_history)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, "
                  f"Team A: {team_a_total_reward:.2f}, "
                  f"Team B: {team_b_total_reward:.2f}, "
                  f"Combined: {combined_performance:.2f}, "
                  f"Difficulty: {curriculum.current_level}")
        
        if combined_performance > best_combined_performance:
            best_combined_performance = combined_performance
            team_a_algorithm.save_models('best_team_a_model.pth')
            team_b_algorithm.save_model('best_team_b_model.pth')
    
    visualizer = VisualizationUtils()
    
    team_a_rewards = [ep['team_a_reward'] for ep in training_history]
    team_b_rewards = [ep['team_b_reward'] for ep in training_history]
    combined_performance = [ep['combined_performance'] for ep in training_history]
    difficulty_levels = [ep['difficulty_level'] for ep in training_history]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(team_a_rewards, label='Team A (MAPPO)', color='red')
    axes[0, 0].set_title('Team A Performance')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].plot(team_b_rewards, label='Team B (QMIX)', color='blue')
    axes[0, 1].set_title('Team B Performance')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Total Reward')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].plot(combined_performance, label='Combined Performance', color='green')
    axes[1, 0].set_title('Overall Performance')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Combined Reward')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].plot(difficulty_levels, label='Difficulty Level', color='purple')
    axes[1, 1].set_title('Curriculum Difficulty')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Difficulty Level')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTraining completed. Best combined performance: {best_combined_performance:.2f}")
    print(f"Final curriculum level: {curriculum.current_level}")
    
    return team_a_algorithm, team_b_algorithm, training_history

if __name__ == "__main__":
    team_a_algo, team_b_algo, history = mixed_training_example()