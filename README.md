# InforMARL Bottleneck Environment

Multi-Agent Reinforcement Learning implementation based on the InforMARL paper for bottleneck navigation scenarios.

## Paper Reference

**"Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation"**  
*Siddharth Nayak, Kenneth Choi, Wenqi Ding, Sydney Dolan, Karthik Gopalakrishnan, Hamsa Balakrishnan*  
ICML 2023 | [arXiv:2211.02127](https://arxiv.org/abs/2211.02127)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install package with make
make install

# Install with development dependencies  
make install-dev
```

## Usage Guide

### 🚀 Quick Start (Easiest Methods)

```bash
# 🎯 Method 1: Using Makefile (Recommended)
make train        # Training with YAML config settings
make demo         # Animation demo
make test         # Quick movement test

# 🎯 Method 2: Direct Python execution
python main.py    # Training (uses configs/env.yaml settings)
python main.py demo    # Animation demo
python main.py test    # Quick movement test
```

### ⚙️ Configuration-Based Usage

The program automatically reads settings from YAML files in `configs/` directory:

```bash
# 1. Edit configuration files
# configs/env.yaml     - Environment settings (agents, map size, etc.)
# configs/model.yaml   - Neural network architecture  
# configs/train.yaml   - Training hyperparameters

# 2. Run with your settings
python main.py        # 🔥 Your YAML settings are automatically applied!
```

**Example: Change number of agents**
```yaml
# Edit configs/env.yaml
environment:
  num_agents: 10      # Change from 6 to 10 agents
  bottleneck_width: 2.0   # Make bottleneck wider
```
```bash
python main.py        # Now runs with 10 agents!
```

### 🔧 Advanced Usage

#### Makefile Commands
```bash
# Training variants
make train           # Default training (YAML settings)
make train-long      # Extended training (200 episodes, 6 agents)

# Demo variants  
make demo            # Default demo (YAML settings)
make demo-big        # Large demo (6 agents)

# Development
make install         # Install package
make lint           # Code quality check
make format         # Auto-format code
make clean          # Clean build files
make help           # Show all commands
```

#### Direct Module Execution
```bash
# Training module
python -m src.informarl_bneck.cli.train              # Default
python -m src.informarl_bneck.cli.train 300 12      # 300 episodes, 12 agents

# Demo module
python -m src.informarl_bneck.cli.demo               # Default  
python -m src.informarl_bneck.cli.demo 8             # 8 agents demo

# Test module
python -m src.informarl_bneck.cli.quicktest          # Default
python -m src.informarl_bneck.cli.quicktest 4        # 4 agents test
```

#### Custom Python Scripts
```python
# Create your own script (e.g., custom_experiment.py)
from src.informarl_bneck.rl.runner import run_informarl_experiment
from src.informarl_bneck.utils.config import get_env_params

# Load and modify config
config = get_env_params("configs")
config['num_agents'] = 20
config['bottleneck_width'] = 0.8  # Very narrow bottleneck

# Run experiment
results, env = run_informarl_experiment(
    num_episodes=500, 
    num_agents=20, 
    config=config
)
```

## Project Structure

```
src/informarl_bneck/
├── env/                    # Environment components
│   ├── bottleneck_env.py   # Main Gym environment
│   ├── map.py              # Map generation  
│   ├── physics.py          # Physics simulation
│   ├── reward.py           # Reward calculation
│   ├── graph_builder.py    # Graph construction
│   └── render.py           # Visualization
├── models/                 # Neural networks
│   ├── gnn.py              # Graph Neural Networks
│   ├── policy.py           # Actor-Critic networks
│   └── agent.py            # InforMARL agent
├── rl/                     # Reinforcement learning
│   └── runner.py           # Training/evaluation runners
├── utils/                  # Utilities
│   └── types.py            # Data structures
└── cli/                    # Command line interfaces
    ├── train.py            # Training CLI
    ├── demo.py             # Demo CLI
    └── quicktest.py        # Test CLI
```

## Features

- **InforMARL Implementation**: Graph Neural Network for information aggregation
- **Bottleneck Navigation**: Agents navigate through narrow corridors
- **Collision Avoidance**: Sophisticated physics and collision handling
- **Visualization**: Real-time rendering and animation
- **Modular Design**: Clean separation of concerns

## Configuration Files

### 📁 configs/env.yaml - Environment Settings
```yaml
environment:
  num_agents: 6           # Number of agents
  agent_radius: 0.5       # Agent size
  corridor_width: 20.0    # Map width
  corridor_height: 10.0   # Map height
  bottleneck_width: 1.2   # Bottleneck passage width
  bottleneck_position: 10.0  # Bottleneck x-position
  sensing_radius: 3.0     # Agent observation range
  max_timesteps: 300      # Episode length
```

### 🧠 configs/model.yaml - Neural Network Architecture
```yaml
gnn:
  input_dim: 6            # Node feature dimension
  hidden_dim: 64          # GNN hidden size
  num_layers: 1           # GNN depth (1 for InforMARL constraint)
  embedding_size: 8       # Entity embedding size
  use_attention: true     # Enable attention mechanism

actor:
  obs_dim: 6              # Local observation dimension
  agg_dim: 64             # Aggregated info dimension
  action_dim: 4           # Number of actions
  hidden_dim: 64          # Actor network size

critic:
  agg_dim: 64             # Global aggregation dimension
  hidden_dim: 64          # Critic network size
```

### 🎓 configs/train.yaml - Training Hyperparameters
```yaml
training:
  num_episodes: 100       # Training episodes
  learning_rate: 0.001    # Learning rate
  batch_size: 32          # Batch size
  update_frequency: 10    # Update every N steps
  
  # PPO parameters
  ppo_epochs: 4           # PPO optimization epochs
  clip_epsilon: 0.2       # PPO clipping parameter
  
  # GAE parameters
  gamma: 0.99             # Discount factor
  lambda: 0.95            # GAE lambda
  
  # Loss weights
  value_loss_coef: 0.5    # Value function loss weight
  entropy_coef: 0.01      # Entropy bonus weight
  
  # Other
  max_grad_norm: 0.5      # Gradient clipping
  memory_size: 10000      # Experience buffer size
  seed: 42                # Random seed
```

## Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'yaml'**
```bash
pip install pyyaml
```

**ModuleNotFoundError: No module named 'torch_geometric'**
```bash
pip install torch-geometric
```

**FileNotFoundError: configs/env.yaml**
```bash
# Make sure you're in the project root directory
cd C:\Users\82104\Desktop\informarl
python main.py
```

### Performance Tips

**For fewer agents (2-4):**
```yaml
# configs/env.yaml
num_agents: 4
# configs/train.yaml  
batch_size: 16
update_frequency: 5
```

**For many agents (10+):**
```yaml
# configs/env.yaml
num_agents: 12
# configs/train.yaml
batch_size: 64
update_frequency: 15
```

**For stable learning:**
```yaml
# configs/train.yaml
learning_rate: 0.0001   # Lower learning rate
ppo_epochs: 8           # More PPO epochs
clip_epsilon: 0.1       # Conservative clipping
```

**For faster learning (less stable):**
```yaml
# configs/train.yaml
learning_rate: 0.01     # Higher learning rate
ppo_epochs: 2           # Fewer PPO epochs
update_frequency: 5     # More frequent updates
```

## Example Experiments

### Experiment 1: Easy vs Hard Bottleneck
```bash
# Easy bottleneck (wide passage)
# Edit configs/env.yaml: bottleneck_width: 2.0
python main.py

# Hard bottleneck (narrow passage)  
# Edit configs/env.yaml: bottleneck_width: 0.8
python main.py
```

### Experiment 2: Few vs Many Agents
```bash
# Few agents (easier coordination)
# Edit configs/env.yaml: num_agents: 4
python main.py

# Many agents (harder coordination)
# Edit configs/env.yaml: num_agents: 12  
python main.py
```

### Experiment 3: Compare Learning Algorithms
```bash
# Conservative learning
# Edit configs/train.yaml: learning_rate: 0.0001, clip_epsilon: 0.1
python main.py

# Aggressive learning
# Edit configs/train.yaml: learning_rate: 0.01, clip_epsilon: 0.3  
python main.py
```

## Expected Output

When you run the training, you should see output like:
```
📋 설정 로드됨:
   - 에이전트 수: 6
   - 에피소드: 100
   - 병목 폭: 1.2
=== InforMARL 2D 병목 환경 학습 시작 ===
Episode 0: Avg Reward = -45.32, Success Rate = 0.00
Episode 10: Avg Reward = -23.15, Success Rate = 0.33
Episode 20: Avg Reward = -12.44, Success Rate = 0.67
...
Episode 90: Avg Reward = 15.67, Success Rate = 0.83

=== 최종 결과 ===
평균 에피소드 보상: 12.456

학습 완료! 애니메이션으로 결과 확인 (y/n)?
```

After training, if you select 'y', you'll see a real-time animation showing:
- Agents (colored circles) navigating the bottleneck
- Goals (green circles) for each agent
- Collision detection and avoidance behaviors  
- Success/collision statistics

## License

MIT License - see LICENSE file for details.