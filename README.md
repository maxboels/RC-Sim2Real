# RC-Sim2Real

> Robot Control: Simulation to Reality Transfer using Isaac Lab

A robotics simulation and training framework built on NVIDIA Isaac Lab for developing and training custom robot controllers with sim-to-real transfer capabilities.

## 🎯 Project Overview

This project focuses on:
- Developing custom robot configurations for simulation
- Training robust control policies in Isaac Sim
- Transferring learned behaviors from simulation to real hardware
- Building reusable components for robot control research

## 🏗️ Project Structure

```
RC-Sim2Real/
├── isaac_lab/           # Isaac Lab fork with custom robots (separate git repo)
│   └── source/
│       └── isaaclab_assets/
│           └── robots/  # Custom robot configurations
├── env_isaaclab/       # Python virtual environment (gitignored)
├── docs/               # Project documentation
├── scripts/            # Helper scripts and utilities
└── README.md          # This file
```

## 🚀 Setup

### Prerequisites

- Ubuntu 24.04+ (GLIBC 2.35+)
- NVIDIA GPU with CUDA 12.8 support
- Python 3.11+
- UV or pip for package management

### Installation

1. **Clone this repository:**
   ```bash
   cd /home/maxboels/projects
   # You're already here!
   ```

2. **Activate the virtual environment:**
   ```bash
   source env_isaaclab/bin/activate
   ```

3. **Isaac Lab is already installed** at `isaac_lab/`
   - This is a fork of [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)
   - Your fork: [maxboels/IsaacLab](https://github.com/maxboels/IsaacLab)

### Verify Installation

```bash
cd isaac_lab
source ../env_isaaclab/bin/activate

# Test Isaac Sim
python -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); app.close()"

# Run Isaac Lab example
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

## 🤖 Developing Custom Robots

### Adding a New Robot

1. Navigate to the robots directory:
   ```bash
   cd isaac_lab/source/isaaclab_assets/isaaclab_assets/robots/
   ```

2. Create a new Python file for your robot configuration (e.g., `my_robot.py`)

3. Define the robot using Isaac Lab's articulation configuration system

4. Register the robot in `__init__.py`

5. Test your robot in simulation

### Example Robot Structure

See existing robots in `isaac_lab/source/isaaclab_assets/isaaclab_assets/robots/` for reference:
- `franka.py` - Franka Emika Panda arm
- `unitree.py` - Unitree quadrupeds
- `quadcopter.py` - Aerial vehicles

## 🎓 Training Workflows

### RL Training

Training scripts use the integrated RL frameworks:
- **rl_games** - High-performance PPO implementation
- **rsl_rl** - ETH Zurich's RL library
- **stable-baselines3** - Popular RL algorithms
- **skrl** - PyTorch RL library

### Sim-to-Real Transfer

Best practices for sim-to-real transfer:
1. Domain randomization in simulation
2. Realistic physics parameters
3. Sensor noise modeling
4. Gradual real-world deployment

## 📚 Documentation

- [Isaac Lab Setup Guide](isaac_lab/SETUP_README.md) - Detailed installation steps
- [Isaac Lab Official Docs](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/)

## 🛠️ Useful Commands

```bash
# Activate environment
source env_isaaclab/bin/activate

# Run Isaac Lab with specific python script
cd isaac_lab
./isaaclab.sh -p path/to/script.py

# Update Isaac Lab extensions
./isaaclab.sh --install

# Format code
./isaaclab.sh --format

# Run tests
./isaaclab.sh --test
```

## 📦 Installed Versions

- **Isaac Sim**: 5.0.0
- **Isaac Lab**: 0.47.1
- **PyTorch**: 2.7.0+cu128
- **Python**: 3.11.14
- **Warp**: 1.9.1

## 🔧 Development Workflow

### Working on Custom Robots

```bash
cd isaac_lab
git checkout -b feature/my-robot-name

# Make changes to source/isaaclab_assets/isaaclab_assets/robots/
# ... develop and test ...

git add .
git commit -m "Add my robot configuration"
git push origin feature/my-robot-name
```

### Updating from Isaac Lab Upstream (Optional)

If you want to pull updates from NVIDIA's Isaac Lab:

```bash
cd isaac_lab
git fetch upstream
git merge upstream/main
git push origin main
```

## 🎯 Current Focus

- [ ] Develop custom robot configurations
- [ ] Create training environments
- [ ] Implement baseline RL policies
- [ ] Test sim-to-real transfer
- [ ] Document deployment procedures

## 📝 Notes

- The `isaac_lab/` directory is managed as a separate git repository (your fork)
- The `env_isaaclab/` virtual environment is specific to this project setup
- Keep documentation updated in the `docs/` folder
- Use `scripts/` for helper utilities and automation

## 🤝 Contributing

This is a personal research project. For contributing to Isaac Lab itself, see the [official repository](https://github.com/isaac-sim/IsaacLab).

## 📄 License

- Isaac Lab: See [isaac_lab/LICENSE](isaac_lab/LICENSE)
- Custom components: [Your chosen license]

---

**Author**: maxboels  
**Last Updated**: October 20, 2025
