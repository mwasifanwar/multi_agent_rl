import numpy as np
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class EvaluationUtils:
    def __init__(self):
        pass
    
    @staticmethod
    def compute_cooperation_metrics(episode_data: Dict[str, Any]) -> Dict[str, float]:
        metrics = {}
        
        rewards = episode_data['rewards']
        actions = episode_data['actions']
        
        total_reward = sum([sum(agent_rewards) for agent_rewards in rewards.values()])
        metrics['total_episode_reward'] = total_reward
        
        agent_rewards = [sum(rewards[agent_id]) for agent_id in rewards.keys()]
        metrics['average_agent_reward'] = np.mean(agent_rewards)
        metrics['std_agent_reward'] = np.std(agent_rewards)
        metrics['fairness_index'] = 1 - (np.std(agent_rewards) / (np.mean(agent_rewards) + 1e-8))
        
        action_coordination = EvaluationUtils._compute_action_coordination(actions)
        metrics['action_coordination'] = action_coordination
        
        return metrics
    
    @staticmethod
    def _compute_action_coordination(actions: Dict[str, List[int]]) -> float:
        if len(actions) == 0:
            return 0.0
        
        timesteps = len(actions[list(actions.keys())[0]])
        coordination_scores = []
        
        for t in range(timesteps):
            timestep_actions = [actions[agent_id][t] for agent_id in actions.keys()]
            
            action_counts = {}
            for action in timestep_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            max_count = max(action_counts.values())
            coordination = max_count / len(timestep_actions)
            coordination_scores.append(coordination)
        
        return np.mean(coordination_scores) if coordination_scores else 0.0
    
    @staticmethod
    def compute_learning_curves(training_history: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        curves = {
            'episode_rewards': [],
            'average_returns': [],
            'cooperation_scores': [],
            'exploration_rates': []
        }
        
        for episode_data in training_history:
            curves['episode_rewards'].append(episode_data.get('total_reward', 0))
            curves['average_returns'].append(episode_data.get('average_return', 0))
            curves['cooperation_scores'].append(episode_data.get('cooperation_score', 0))
            curves['exploration_rates'].append(episode_data.get('exploration_rate', 0))
        
        return curves
    
    @staticmethod
    def compute_performance_metrics(evaluation_results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not evaluation_results:
            return {}
        
        metrics = {}
        
        total_rewards = [result.get('total_reward', 0) for result in evaluation_results]
        cooperation_scores = [result.get('cooperation_score', 0) for result in evaluation_results]
        success_rates = [result.get('success_rate', 0) for result in evaluation_results]
        
        metrics['mean_total_reward'] = np.mean(total_rewards)
        metrics['std_total_reward'] = np.std(total_rewards)
        metrics['mean_cooperation_score'] = np.mean(cooperation_scores)
        metrics['mean_success_rate'] = np.mean(success_rates)
        metrics['efficiency'] = metrics['mean_total_reward'] / max(1, len(evaluation_results))
        
        return metrics
    
    @staticmethod
    def analyze_emergent_behavior(episode_data: Dict[str, Any]) -> Dict[str, Any]:
        analysis = {}
        
        actions = episode_data.get('actions', {})
        states = episode_data.get('states', {})
        
        if not actions or not states:
            return analysis
        
        analysis['strategy_patterns'] = EvaluationUtils._identify_strategy_patterns(actions)
        analysis['role_specialization'] = EvaluationUtils._analyze_role_specialization(actions, states)
        analysis['communication_patterns'] = EvaluationUtils._analyze_communication_patterns(actions)
        
        return analysis
    
    @staticmethod
    def _identify_strategy_patterns(actions: Dict[str, List[int]]) -> Dict[str, float]:
        patterns = {}
        
        for agent_id, agent_actions in actions.items():
            action_sequence = ''.join(map(str, agent_actions))
            
            unique_actions = len(set(agent_actions))
            patterns[f'{agent_id}_diversity'] = unique_actions / len(agent_actions) if agent_actions else 0
            
            if len(agent_actions) >= 3:
                repeated_patterns = 0
                for i in range(len(agent_actions) - 2):
                    if agent_actions[i] == agent_actions[i+1] == agent_actions[i+2]:
                        repeated_patterns += 1
                patterns[f'{agent_id}_consistency'] = repeated_patterns / (len(agent_actions) - 2)
        
        return patterns
    
    @staticmethod
    def _analyze_role_specialization(actions: Dict[str, List[int]], states: Dict[str, List[Any]]) -> Dict[str, float]:
        specialization = {}
        
        if not actions or not states:
            return specialization
        
        action_distributions = {}
        for agent_id, agent_actions in actions.items():
            action_counts = np.bincount(agent_actions, minlength=10)
            action_distributions[agent_id] = action_counts / len(agent_actions) if agent_actions else np.zeros(10)
        
        action_matrix = np.array(list(action_distributions.values()))
        
        if len(action_matrix) > 1:
            specialization_score = 1 - np.mean(np.corrcoef(action_matrix))
            specialization['overall_specialization'] = max(0, specialization_score)
        
        return specialization
    
    @staticmethod
    def _analyze_communication_patterns(actions: Dict[str, List[int]]) -> Dict[str, float]:
        communication = {}
        
        if len(actions) < 2:
            return communication
        
        timesteps = len(actions[list(actions.keys())[0]])
        response_correlations = []
        
        for t in range(1, timesteps):
            current_actions = [actions[agent_id][t] for agent_id in actions.keys()]
            previous_actions = [actions[agent_id][t-1] for agent_id in actions.keys()]
            
            if len(set(previous_actions)) > 1 and len(set(current_actions)) > 1:
                correlation = np.corrcoef(previous_actions, current_actions)[0, 1]
                if not np.isnan(correlation):
                    response_correlations.append(abs(correlation))
        
        communication['response_correlation'] = np.mean(response_correlations) if response_correlations else 0.0
        
        return communication