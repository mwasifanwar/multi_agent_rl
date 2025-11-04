<!DOCTYPE html>
<html>
<head>
</head>
<body>

<h1>Multi-Agent Reinforcement Learning for Complex Cooperative and Competitive Systems</h1>

<p>A comprehensive framework for advanced multi-agent reinforcement learning research, implementing state-of-the-art algorithms for complex cooperative, competitive, and mixed scenarios.</p>

<h2>Overview</h2>

<p>This project provides a sophisticated multi-agent reinforcement learning (MARL) framework that enables research and development of intelligent agent systems capable of learning complex coordinated behaviors. The framework implements cutting-edge MARL algorithms including MADDPG, QMIX, IQL, and MAPPO, with support for both centralized and decentralized training paradigms.</p>

<p>Key objectives include:</p>
<ul>
  <li>Enabling research in emergent multi-agent behaviors</li>
  <li>Providing scalable implementations of state-of-the-art MARL algorithms</li>
  <li>Supporting both cooperative and competitive multi-agent scenarios</li>
  <li>Facilitating curriculum learning and complex environment design</li>
  <li>Offering comprehensive evaluation and visualization tools</li>
</ul>

<img width="797" height="513" alt="image" src="https://github.com/user-attachments/assets/f0dadcc0-b6b5-401b-9f0c-b83b22962373" />


<h2>System Architecture</h2>

<p>The framework follows a modular architecture that separates environment simulation, agent policies, learning algorithms, and evaluation utilities. The core system workflow operates as follows:</p>

<pre><code>
Environment Observation → Agent Policy Networks → Action Selection → Environment Step
       ↑                                                                  ↓
    Replay Buffer ← Experience Storage ← Reward Calculation ← Next Observation
       ↓
Algorithm Update (Policy Gradients, Q-learning, etc.) → Model Improvement
</code></pre>

<img width="1187" height="535" alt="image" src="https://github.com/user-attachments/assets/fd192f56-f5fd-4f0a-ac08-9c08fc0ffbe1" />


<p>The architecture supports both decentralized execution (where agents make decisions based on local observations) and centralized training (where additional global information may be used during learning).</p>

<h2>Technical Stack</h2>

<ul>
  <li><strong>Deep Learning Framework:</strong> PyTorch 1.9+</li>
  <li><strong>Environment Simulation:</strong> OpenAI Gym</li>
  <li><strong>Numerical Computing:</strong> NumPy</li>
  <li><strong>Visualization:</strong> Matplotlib, Seaborn</li>
  <li><strong>Development Tools:</strong> pytest, black, flake8</li>
</ul>

<h2>Mathematical Foundation</h2>

<p>The framework implements several advanced MARL algorithms based on rigorous mathematical foundations:</p>

<h3>Multi-Agent Deep Deterministic Policy Gradient (MADDPG)</h3>
<p>MADDPG extends DDPG to multi-agent settings using centralized critics. The objective function for agent $i$ is:</p>
<p>$J(\theta_i) = \mathbb{E}_{\mathbf{o},\mathbf{a} \sim \mathcal{D}}[Q_i^{\phi}(\mathbf{o}, a_1, \ldots, a_N)]$</p>
<p>where the centralized action-value function $Q_i^{\phi}$ takes all agents' actions as input.</p>

<h3>QMIX</h3>
<p>QMIX employs monotonic value function factorization to ensure individual action selection consistency with joint action-value maximization:</p>
<p>$\frac{\partial Q_{tot}}{\partial Q_a} \geq 0 \quad \forall a \in A$</p>
<p>The mixing network ensures this monotonicity constraint while allowing rich representation of the joint action-value function.</p>

<h3>Multi-Agent PPO (MAPPO)</h3>
<p>MAPPO extends PPO to multi-agent settings with the clipped objective:</p>
<p>$L^{CLIP}(\theta) = \mathbb{E}[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$</p>
<p>where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ and $\hat{A}_t$ is the advantage estimate.</p>

<h2>Features</h2>

<ul>
  <li><strong>Multiple Environment Types:</strong> Cooperative navigation, predator-prey, and custom mixed scenarios</li>
  <li><strong>Advanced Algorithms:</strong> MADDPG, QMIX, IQL, MAPPO with centralized and decentralized variants</li>
  <li><strong>Flexible Architecture:</strong> Support for both cooperative and competitive multi-agent learning</li>
  <li><strong>Comprehensive Training:</strong> Experience replay, prioritized replay, curriculum learning</li>
  <li><strong>Sophisticated Analysis:</strong> Cooperation metrics, emergence analysis, performance evaluation</li>
  <li><strong>Visualization Tools:</strong> Learning curves, agent performance, action distributions</li>
  <li><strong>Modular Design:</strong> Easy extension with new algorithms and environments</li>
</ul>

<img width="736" height="429" alt="image" src="https://github.com/user-attachments/assets/be00d62b-60d8-47b5-807e-f23a1036bed5" />


<h2>Installation</h2>

<p>Follow these steps to set up the environment and install dependencies:</p>

<pre><code>
# Clone the repository
git clone https://github.com/mwasifanwar/multi-agent-rl.git
cd multi-agent-rl

# Create a virtual environment (recommended)
python -m venv marl_env
source marl_env/bin/activate  # On Windows: marl_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Verify installation
python -c "import multi_agent_rl; print('Installation successful!')"
</code></pre>

<h2>Usage / Running the Project</h2>

<h3>Basic Training Example</h3>
<pre><code>
# Train cooperative agents using MADDPG
python main.py --mode cooperative --algorithm MADDPG --episodes 1000 --agents 3

# Train competitive predator-prey scenario
python main.py --mode competitive --episodes 1500

# Run mixed cooperative-competitive training
python main.py --mode mixed --episodes 2000
</code></pre>

<h3>Advanced Usage with Custom Configuration</h3>
<pre><code>
import torch
from multi_agent_rl.core import CooperativeNavigationEnv
from multi_agent_rl.algorithms import MADDPG

# Initialize environment and algorithm
env = CooperativeNavigationEnv(num_agents=3, state_dim=20, action_dim=5)
algorithm = MADDPG(num_agents=3, state_dim=20, action_dim=5)

# Training loop
for episode in range(1000):
    observations = env.reset()
    episode_rewards = {f'agent_{i}': 0 for i in range(3)}
    
    for step in range(200):
        actions = algorithm.select_actions(observations, training=True)
        next_observations, rewards, dones, infos = env.step(actions)
        
        # Store experience and update
        # ... (complete training logic)
        
        observations = next_observations
        
        if all(dones.values()):
            break
    
    # Periodic evaluation and model saving
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {sum(episode_rewards.values()):.2f}")
</code></pre>

<h2>Configuration / Parameters</h2>

<h3>Algorithm Hyperparameters</h3>
<ul>
  <li><strong>Learning Rates:</strong> Actor (0.001), Critic (0.001)</li>
  <li><strong>Discount Factor (γ):</strong> 0.95-0.99</li>
  <li><strong>Replay Buffer Size:</strong> 10,000-20,000 experiences</li>
  <li><strong>Batch Size:</strong> 256-512</li>
  <li><strong>Exploration:</strong> ε-greedy with decay from 1.0 to 0.01</li>
  <li><strong>Target Network Update (τ):</strong> 0.01 for soft updates</li>
</ul>

<h3>Environment Parameters</h3>
<ul>
  <li><strong>State Dimensions:</strong> 10-25 depending on scenario complexity</li>
  <li><strong>Action Space:</strong> Discrete (4-6 actions) or continuous</li>
  <li><strong>Episode Length:</strong> 200-300 steps</li>
  <li><strong>Agent Count:</strong> 2-6 agents for different scenarios</li>
</ul>

<h2>Folder Structure</h2>

<pre><code>
multi_agent_rl/
├── core/                          # Core framework components
│   ├── __init__.py
│   ├── multi_agent_env.py         # Base environment class
│   ├── agent_manager.py           # Agent coordination and management
│   ├── policy_network.py          # Neural network architectures
│   ├── value_network.py           # Value function approximators
│   └── experience_replay.py       # Replay buffer implementations
├── algorithms/                    # MARL algorithm implementations
│   ├── __init__.py
│   ├── maddpg.py                  # MADDPG algorithm
│   ├── qmix.py                    # QMIX algorithm
│   ├── iql.py                     # Independent Q-learning
│   └── mappo.py                   # Multi-agent PPO
├── environments/                  # Custom environment implementations
│   ├── __init__.py
│   ├── cooperative_navigation.py  # Cooperative multi-agent navigation
│   └── predator_prey.py           # Competitive predator-prey scenario
├── utils/                         # Utility functions and tools
│   ├── __init__.py
│   ├── training_utils.py          # Training helpers and curriculum learning
│   ├── evaluation_utils.py        # Performance metrics and analysis
│   └── visualization_utils.py     # Plotting and visualization
├── examples/                      # Example training scripts
│   ├── __init__.py
│   ├── cooperative_training.py    # Cooperative scenario examples
│   ├── competitive_training.py    # Competitive scenario examples
│   └── mixed_training.py          # Mixed scenario examples
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation script
└── main.py                        # Main entry point
</code></pre>

<h2>Results / Experiments / Evaluation</h2>

<h3>Performance Metrics</h3>
<p>The framework includes comprehensive evaluation metrics:</p>
<ul>
  <li><strong>Total Episode Reward:</strong> Sum of all agents' rewards</li>
  <li><strong>Cooperation Score:</strong> Measures coordination between agents (0-1 scale)</li>
  <li><strong>Fairness Index:</strong> Measures reward distribution equality</li>
  <li><strong>Action Coordination:</strong> Temporal alignment of agent actions</li>
  <li><strong>Success Rate:</strong> Task completion frequency</li>
</ul>

<h3>Experimental Results</h3>
<p>Typical training performance across different scenarios:</p>
<ul>
  <li><strong>Cooperative Navigation:</strong> Agents learn to efficiently cover targets with 80-95% success rate</li>
  <li><strong>Predator-Prey:</strong> Predators develop coordinated hunting strategies with 70-85% capture rate</li>
  <li><strong>Mixed Scenarios:</strong> Teams learn both cooperative and competitive behaviors simultaneously</li>
</ul>

<h3>Emergent Behavior Analysis</h3>
<p>The framework enables analysis of complex emergent behaviors:</p>
<ul>
  <li><strong>Role Specialization:</strong> Agents spontaneously develop specialized roles</li>
  <li><strong>Communication Patterns:</strong> Implicit communication through action sequences</li>
  <li><strong>Strategy Evolution:</strong> Progressive development of sophisticated multi-step strategies</li>
</ul>

<h2>References / Citations</h2>

<ol>
  <li>Lowe, R., et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." <em>NeurIPS</em> 2017. <a href="https://arxiv.org/abs/1706.02275">arXiv:1706.02275</a></li>
  <li>Rashid, T., et al. "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning." <em>ICML</em> 2018. <a href="https://arxiv.org/abs/1803.11485">arXiv:1803.11485</a></li>
  <li>Yu, C., et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." <em>NeurIPS</em> 2021. <a href="https://arxiv.org/abs/2103.01955">arXiv:2103.01955</a></li>
  <li>Tan, M. "Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents." <em>ICML</em> 1993.</li>
  <li>Foerster, J., et al. "Counterfactual Multi-Agent Policy Gradients." <em>AAAI</em> 2018. <a href="https://arxiv.org/abs/1705.08926">arXiv:1705.08926</a></li>
</ol>

<h2>Acknowledgements</h2>

<p>This project builds upon foundational research in multi-agent reinforcement learning and leverages several open-source libraries:</p>
<ul>
  <li>PyTorch team for the deep learning framework</li>
  <li>OpenAI for the Gym environment interface</li>
  <li>The multi-agent RL research community for algorithm development</li>
  <li>Contributors to the numerical computing and visualization ecosystems</li>
</ul>

<br>

<h2 align="center">✨ Author</h2>

<p align="center">
  <b>M Wasif Anwar</b><br>
  <i>AI/ML Engineer | Effixly AI</i>
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/mwasifanwar" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn">
  </a>
  <a href="mailto:wasifsdk@gmail.com">
    <img src="https://img.shields.io/badge/Email-grey?style=for-the-badge&logo=gmail" alt="Email">
  </a>
  <a href="https://mwasif.dev" target="_blank">
    <img src="https://img.shields.io/badge/Website-black?style=for-the-badge&logo=google-chrome" alt="Website">
  </a>
  <a href="https://github.com/mwasifanwar" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
</p>

<br>

---

<div align="center">

### ⭐ Don't forget to star this repository if you find it helpful!

</div>

</body>
</html>
