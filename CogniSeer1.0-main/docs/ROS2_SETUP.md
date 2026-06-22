# ROS2 Setup and Dependency Guide

## Overview

ROS2 (Robot Operating System 2) is an **optional dependency** for this project. It is required only for deploying the robot brain on actual ROS2-enabled robots. All training, export, and testing functionality works independently of ROS2.

### When You Need ROS2

| Use Case | Requires ROS2? | Notes |
|----------|---|---|
| **Train the model** | ❌ No | `python train_brain.py` or `cogniseer-train` |
| **Export/quantize model** | ❌ No | `python quantize_brain.py` or `cogniseer-export` |
| **Run smoke tests** | ❌ No | `make smoke` or `cogniseer-smoke` |
| **Deploy on ROS2 robot** | ✅ **Yes** | Build `robot_brain_ros2` or `robot_brain_stack` packages |
| **Develop ROS2 components** | ✅ **Yes** | Work with `ros2_robot_brain_node.py` or ROS2 workspace |

### When You Don't Need ROS2

- **Local development** of training, inference, safety layers, or data pipelines
- **Docker-based CI/CD** for model training and testing
- **Edge devices without ROS2** (using `edge_runtime.py` directly)

---

## ROS2 Installation

### Prerequisite: Ubuntu/Debian Linux

ROS2 requires Linux. Official support is for:
- **Ubuntu 22.04 LTS** (ROS2 Humble, recommended)
- **Ubuntu 24.04 LTS** (ROS2 Iron)

### Install ROS2 Humble (Recommended)

Follow the official installation guide: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

**Quick summary**:

```bash
# 1. Set locale (UTF-8)
locale  # Verify en_US.UTF-8 is available
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
export LANG=en_US.UTF-8

# 2. Add ROS2 GPG key and repository
sudo apt update
sudo apt install curl -y
sudo curl -sSL https://repo.ros2.org/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros2.org/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 3. Install ROS2 (desktop version recommended)
sudo apt update
sudo apt install ros-humble-desktop -y

# 4. Setup environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# 5. Install build tools
sudo apt install ros-humble-ros-core -y
sudo apt install python3-colcon-common-extensions -y
```

### Verify Installation

```bash
# Activate ROS2 environment
source /opt/ros/humble/setup.bash

# Test ROS2
ros2 --version
# Expected output: ROS 2 Humble Hawksbill (version tag)

# Check ROS2 Python packages
python3 -c "import rclpy; print('✓ rclpy available')"
```

---

## Environment Setup

### One-Time Setup

1. **Install this project** (if not already done):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.lock
   pip install -e .
   ```

2. **Activate ROS2 environment**:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

   **Tip**: Add to `~/.bashrc` to auto-activate ROS2:
   ```bash
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   ```

### Every Session

Before working with ROS2 components, activate both environments:

```bash
# 1. Activate Python virtual environment
source /path/to/project/.venv/bin/activate

# 2. Source ROS2 setup
source /opt/ros/humble/setup.bash

# Verify both are active
which python  # Should point to .venv
which ros2    # Should find ROS2 command
echo $ROS_DISTRO  # Should print "humble"
```

### Docker Alternative (For CI/CD)

If deploying via Docker:

```dockerfile
FROM ros:humble

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Install project
COPY . /workspace/
WORKDIR /workspace
RUN python3 -m venv .venv
RUN .venv/bin/pip install -r requirements.lock
RUN .venv/bin/pip install -e .
```

---

## Dependency Resolution

### Case 1: ROS2 Not Installed (Development/Training)

If you try to use ROS2-specific code without ROS2 installed, you'll get a clear error:

```python
>>> import ros2_robot_brain_node
ImportError: ROS2 is not installed or not available in this Python environment.
This module (ros2_robot_brain_node.py) requires ROS2.
Install ROS2 according to https://docs.ros.org/en/humble/Installation.html,
then activate the ROS2 environment before using this module.
For standalone training/export without ROS2, use train_brain.py and quantize_brain.py instead.
```

**Solution**: Use only the non-ROS2 modules:
- ✅ `train_brain.py` — Train models
- ✅ `quantize_brain.py` — Export and quantize
- ✅ `edge_runtime.py` — Inference (standalone)
- ✅ `safety_layer.py` — Safety validation
- ❌ `ros2_robot_brain_node.py` — Requires ROS2

### Case 2: ROS2 Installed, Project Virtual Env Not Activated

Error:
```
ModuleNotFoundError: No module named 'torch'
```

**Solution**: Ensure both environments are activated:
```bash
source .venv/bin/activate
source /opt/ros/humble/setup.bash
```

### Case 3: ROS2 Installed, But Python Packages Not Found

Error:
```
ModuleNotFoundError: No module named 'rclpy'
```

**Solution**: ROS2 provides system Python packages. They may not be in the virtual env's Python path.

**Option A**: Use ROS2's Python directly (avoid venv):
```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 -m pip install -e .
/usr/bin/python3 ros2_robot_brain_node.py
```

**Option B**: Install ROS2 packages in venv (workaround):
```bash
source .venv/bin/activate
source /opt/ros/humble/setup.bash

# Copy ROS2 packages to venv (risky, not recommended)
cp -r /opt/ros/humble/lib/python3.10/dist-packages/* $VIRTUAL_ENV/lib/python3.10/site-packages/
```

**Option C** (Recommended): Build within ROS2 workspace:
```bash
# Create ROS2 workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone <this-repo> cogni-sim-brain
cd ~/ros2_ws
colcon build
source install/setup.bash
```

---

## ROS2 Workspace Integration

For production deployment, integrate this project into a ROS2 workspace:

### Structure

```
~/robot_ws/                         # ROS2 workspace root
├── src/
│   ├── robot_brain_stack/          # Main ROS2 brain package (colcon build)
│   └── other-robot-packages/       # Your other ROS2 packages
├── build/                          # Build outputs (auto-generated)
├── install/                        # Installation (auto-generated)
└── log/                            # Build logs (auto-generated)
```

### Build & Run

```bash
cd ~/robot_ws

# Build all packages (including robot_brain_stack)
colcon build

# Source the install overlay
source install/setup.bash

# Launch the brain stack
ros2 launch robot_brain_stack brain_stack.launch.py

# Or run individual node
ros2 run robot_brain_stack robot_brain_node
```

---

## Isolated Development (No ROS2 Required)

For pure ML/training development **without ROS2**:

```bash
# Standard installation (no ROS2)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
pip install -e .

# All training/inference commands work
python train_brain.py
python quantize_brain.py
cogniseer-smoke --smoke-steps 50
```

**What's available**:
- ✅ All data pipeline utilities
- ✅ SNN training and export
- ✅ Quantization and benchmarking
- ✅ Safety layer testing
- ✅ Edge runtime inference

**What's NOT available**:
- ❌ `ros2_robot_brain_node` (requires ROS2)
- ❌ ROS2-specific launch files
- ❌ ROS2 message/topic integration

---

## Deployment Checklist

Before deploying on a ROS2 robot:

- [ ] ROS2 is installed (`ros2 --version`)
- [ ] ROS2 environment is sourced (`echo $ROS_DISTRO` → `humble`)
- [ ] Project is installed in ROS2 workspace or venv with ROS2 available
- [ ] `import rclpy` works: `python3 -c "import rclpy"`
- [ ] Model artifacts are built: `ls -la deploy/robot_brain_int8_stateful.pt`
- [ ] Configuration is correct: `cat config/prod.json | head -20`
- [ ] ROS2 node starts: `ros2 run robot_brain_ros2 robot_brain_node`

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'rclpy'"

**Cause**: ROS2 environment not sourced, or ROS2 not installed.

**Solution**:
```bash
# Check ROS2 installation
which ros2
source /opt/ros/humble/setup.bash
echo $ROS_DISTRO

# Verify rclpy is available
python3 -c "import rclpy; print(rclpy.__file__)"
```

### Issue: "Cannot find ROS2 workspace"

**Cause**: Not inside a `robot_ws` with colcon-built packages.

**Solution**:
```bash
# If using robot_ws workspace
source ~/robot_ws/install/setup.bash

# If using venv with system ROS2
source .venv/bin/activate
source /opt/ros/humble/setup.bash
```

### Issue: "ROS2 packages built with Python 3.10, but venv uses Python 3.12"

**Cause**: Python version mismatch between ROS2 and virtual environment.

**Solution**:
```bash
# Create venv with same Python as ROS2
python3.10 -m venv .venv
source .venv/bin/activate
source /opt/ros/humble/setup.bash
pip install -e .
```

### Issue: "ros2 launch fails with package not found"

**Cause**: Workspace not rebuilt after installing project.

**Solution**:
```bash
cd ~/robot_ws
colcon build --symlink-install
source install/setup.bash
```

---

## CI/CD Integration

### GitHub Actions (Training Without ROS2)

```yaml
- name: Install dependencies
  run: |
    pip install -r requirements.lock
    pip install -e .

- name: Train model
  run: python train_brain.py

- name: Smoke test
  run: cogniseer-smoke
```

### GitHub Actions (ROS2 Deployment)

```yaml
container:
  image: ros:humble

- name: Install dependencies
  run: |
    apt-get update && apt-get install -y python3-pip
    pip install -r requirements.lock
    pip install -e .

- name: Build ROS2 workspace
  run: |
    mkdir -p ~/ros2_ws/src
    cp -r . ~/ros2_ws/src/
    cd ~/ros2_ws
    colcon build

- name: Test ROS2 node
  run: |
    source ~/ros2_ws/install/setup.bash
    ros2 run robot_brain_ros2 robot_brain_node --help
```

---

## References

- **Official ROS2 Documentation**: https://docs.ros.org/
- **ROS2 Humble Installation**: https://docs.ros.org/en/humble/Installation.html
- **ROS2 Python Client Library**: https://docs.ros.org/en/humble/Concepts/About-ROS-2-Client-Libraries.html
- **colcon build system**: https://colcon.readthedocs.io/

---

## Summary

| Scenario | ROS2 Required? | Installation |
|----------|---|---|
| Local training | ❌ No | `pip install -e .` |
| CI/CD training | ❌ No | Ubuntu container, no ROS2 layer |
| Local ROS2 development | ✅ Yes | `apt install ros-humble-desktop` + venv |
| Production robot deployment | ✅ Yes | Full ROS2 workspace + `colcon build` |

**Recommended**: Use ROS2 only when deploying to actual robots. Keep development environment lightweight without ROS2 for faster iteration.
