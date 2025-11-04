import argparse
import numpy as np
import torch
from examples.cooperative_training import cooperative_training_example
from examples.competitive_training import competitive_training_example
from examples.mixed_training import mixed_training_example

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning Framework")
    parser.add_argument('--mode', choices=['cooperative', 'competitive', 'mixed', 'demo'], default='demo',
                       help='Training mode: cooperative, competitive, mixed, or demo')
    parser.add_argument('--algorithm', choices=['MADDPG', 'QMIX', 'IQL', 'MAPPO'], default='MADDPG',
                       help='RL algorithm to use')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--agents', type=int, default=3, help='Number of agents')
    parser.add_argument('--render', action='store_true', help='Render environment during training')
    
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    if args.mode == 'cooperative':
        print("Starting cooperative training...")
        cooperative_training_example()
        
    elif args.mode == 'competitive':
        print("Starting competitive training...")
        competitive_training_example()
        
    elif args.mode == 'mixed':
        print("Starting mixed training...")
        mixed_training_example()
        
    else:
        print("Running demonstration...")
        from examples.cooperative_training import cooperative_training_example
        cooperative_training_example()

if __name__ == "__main__":
    main()