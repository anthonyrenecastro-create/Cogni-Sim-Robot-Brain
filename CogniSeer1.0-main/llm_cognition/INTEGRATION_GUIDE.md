# LLM Cognition Layer Integration Guide

## Adding AI Reasoning to CogniSeer Robot Brain

This guide shows how to integrate the new LLM-based slow brain into the existing CogniSeer architecture.

## Architecture Overview

### Before (Fast Brain Only)
```
PyTorch Fast Brain (20 Hz)
  ├─ Obstacle avoidance
  ├─ Emergency stop
  ├─ Velocity control
  └─ Local navigation
```

### After (Fast + Slow Brain)
```
┌─ PyTorch Fast Brain (20 Hz)
│  ├─ Obstacle avoidance
│  ├─ Emergency stop
│  ├─ Velocity control
│  └─ Local navigation
│
└─ LLM Slow Brain (0.1-1 Hz, async)
   ├─ Mission planning
   ├─ Recovery strategy selection
   ├─ Natural language explanations
   └─ Learning from experience
```

## Integration Steps

### Step 1: Install Dependencies

```bash
cd CogniSeer1.0-main

# Core dependencies
pip install requests numpy

# For Ollama (recommended for edge)
# No additional Python packages needed, just install Ollama server

# For OpenAI (if you prefer cloud)
pip install openai tiktoken

# For testing
pip install pytest pytest-asyncio
```

### Step 2: Set Up Local LLM (Ollama)

```bash
# 1. Install Ollama
# Download from https://ollama.ai/
# or: curl https://ollama.ai/install.sh | sh

# 2. Start Ollama server
ollama serve  # Runs on localhost:11434

# 3. Pull a model (in another terminal)
ollama pull llama2              # 7B, fast, good for planning
ollama pull mistral             # Better reasoning
ollama pull neural-chat         # Good conversational ability
```

### Step 3: Integrate with Existing Brain

#### Modify `robot_brain_ros2_node.py`

Add slow brain alongside fast brain:

```python
from llm_cognition import CognitionLayer
from llm_cognition.decision_router import DecisionType, Urgency
import asyncio

class RobotBrainNode(Node):
    def __init__(self):
        super().__init__('robot_brain_node')
        
        # Existing fast brain
        self.fast_brain = self._load_pytorch_model()
        
        # NEW: Add slow brain for planning/recovery
        self.slow_brain = CognitionLayer(
            provider="ollama",
            model="llama2",
            memory_persistence_dir="./robot_brain_memory"
        )
        
        # Timer for fast brain (20 Hz)
        self.create_timer(0.05, self.fast_brain_loop)
        
        # Timer for slow brain (1 Hz)
        self.create_timer(1.0, self.slow_brain_loop)
    
    def fast_brain_loop(self):
        """Fast brain: PyTorch control at 20 Hz."""
        # Get observation from sensor aggregator
        observation = self.get_latest_observation()
        
        # Fast neural network decision
        fast_decision = self.fast_brain(observation)
        
        # Route through hybrid router
        decision = self.slow_brain.decide(
            observation=observation,
            decision_type=DecisionType.OBSTACLE_AVOIDANCE,
            urgency=Urgency.HIGH,
            fast_brain_output={
                "action": fast_decision["action"],
                "confidence": fast_decision["confidence"]
            }
        )
        
        # Execute (fast brain usually runs)
        self.execute_decision(decision)
    
    def slow_brain_loop(self):
        """Slow brain: LLM planning at 1 Hz."""
        # Check if we have a goal
        if not self.current_goal:
            return
        
        # Async planning without blocking fast brain
        asyncio.create_task(
            self.slow_brain.plan_mission(self.current_goal)
        )
        
        # Check for failures needing recovery planning
        if self.failure_detected:
            asyncio.create_task(
                self.slow_brain.handle_failure(
                    self.last_failure,
                    context=f"Position: {self.current_position}"
                )
            )
```

#### Modify `sensor_aggregator.py`

Pass sensor observations to cognition layer for context:

```python
# After aggregating sensors into 32D vector
observation_vector = Float32MultiArray(data=normalized_vector)
self.sensor_pub.publish(observation_vector)

# Also update cognition layer's spatial memory
if hasattr(self, 'cognition'):
    self.cognition.memory.spatial.add_feature(
        name=f"obstacle_{time.time()}",
        position=(odometry_x, odometry_y),
        type="obstacle",
        confidence=1.0,
        metadata={"distance": front_distance}
    )
```

### Step 4: Configure Brain Parameters

Create `config/llm_params.yaml`:

```yaml
# LLM Cognition Layer Configuration

llm:
  # Provider selection
  provider: "ollama"  # or "openai", "claude", "huggingface"
  model: "llama2"
  
  # Ollama specific
  ollama:
    base_url: "http://localhost:11434"
    temperature: 0.7
    timeout: 120
  
  # OpenAI specific (if used)
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
  
  # Claude specific (if used)
  claude:
    api_key: "${ANTHROPIC_API_KEY}"

# Memory configuration
memory:
  short_term_size: 10
  enable_long_term: true
  enable_episodic: true
  enable_spatial: true
  persistence_dir: "./robot_memory"

# Decision routing
routing:
  critical_timeout: 0.1  # seconds
  high_timeout: 0.5
  medium_timeout: 5.0
  low_timeout: 30.0
  
  # When to use slow brain
  use_slow_brain_for:
    - planning
    - recovery
    - communication
    - explanation

# Performance
performance:
  fast_brain_frequency: 20  # Hz
  slow_brain_frequency: 1   # Hz
  max_llm_tokens: 1024
```

### Step 5: Launch File Integration

Update `brain_stack.launch.py`:

```python
def generate_launch_description():
    return LaunchDescription([
        # ... existing nodes ...
        
        # NEW: Slow brain node for LLM-based planning
        Node(
            package='robot_brain_ros2',
            executable='slow_brain_node.py',
            name='slow_brain',
            parameters=[
                {'llm_provider': 'ollama'},
                {'llm_model': 'llama2'},
                {'memory_dir': os.path.expanduser('~/robot_memory')},
            ],
            output='screen',
        ),
    ])
```

### Step 6: Test Integration

```bash
# 1. Make sure Ollama is running
ollama serve  # In one terminal

# 2. Run tests
cd llm_cognition/tests
python -m pytest test_cognition_layer.py -v

# 3. Run with robot
ros2 launch robot_brain_stack brain_stack.launch.py
```

## Usage Examples

### Planning a Mission

```python
# In your mission control node
cognition = CognitionLayer(provider="ollama", model="llama2")

# Start mission
mission_id = cognition.start_mission("explore_building_floor_3")

# Async planning (doesn't block control loop)
await cognition.plan_mission(
    goal="explore_building_floor_3",
    context="Building has 3 floors, currently on floor 3"
)

# ... fast brain runs autonomously meanwhile ...

# End mission with result
cognition.end_mission(
    success=True,
    outcome="Successfully explored and mapped entire floor"
)
```

### Handling Recovery

```python
# In your fast brain failure detection
if collision_detected:
    # Async recovery planning
    await cognition.handle_failure(
        failure="collision_with_wall",
        context=f"Robot at {position}, tried moving {direction}"
    )
    
    # Get explanation
    explanation = await cognition.explain_decision(decision)
    print(explanation)
```

### Memory and Learning

```python
# Add observations for learning
cognition.add_observation("Area A has many obstacles")
cognition.add_observation("Door at (5.2, 3.1) is usually locked")

# Query memory for context
context = cognition.memory.get_full_context(
    robot_position=(5.0, 3.0),
    context_radius=10.0  # meters
)

# Get mission history
successful_missions = cognition.memory.episodic.get_successful_missions(
    goal_filter="navigate"
)
```

## Decision Routing Examples

### Fast Brain (Reactive, <100ms)
```
Sensor input → Fast brain NN → Decision
  └─ Obstacle detected at 0.3m → Turn 30° right (immediate)
  └─ Collision risk → Emergency stop (immediate)
  └─ Path clear → Continue forward (immediate)
```

### Slow Brain (Planning, <5s)
```
Goal + Context → LLM → Plan + Explanation
  └─ "Navigate to target" → Multi-step plan with waypoints
  └─ "Stuck on obstacle" → Recovery strategy with fallback
  └─ "User asks why?" → Explanation with reasoning
```

### Hybrid Decision

```python
def make_decision(observation, goal):
    # Fast brain always runs (critical decisions)
    if urgency == CRITICAL:
        return fast_brain.decide(observation)
    
    # Slow brain for complex decisions
    if decision_type == PLANNING:
        return await slow_brain.plan_mission(goal)
    
    # Default: Fast brain
    return fast_brain.decide(observation)
```

## Monitoring and Debugging

### Check Decision Statistics

```python
stats = cognition.get_decision_statistics()
print(f"Total decisions: {stats['total_decisions']}")
print(f"Fast brain: {stats['fast_brain_decisions']}")
print(f"Slow brain: {stats['slow_brain_decisions']}")
print(f"Average confidence: {stats['average_confidence']:.2f}")
```

### Monitor Memory Usage

```bash
# Short-term memory
echo $cognition.memory.short_term.messages | wc -l  # Should be < 10

# Long-term memory
ls -lh robot_memory/long_term.json

# Episodic memory
ls -lh robot_memory/episodic.json
```

### Check Ollama Status

```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Monitor resource usage
top | grep ollama
```

## Performance Tuning

### Reduce Latency

```python
# Use smaller model
cognition = CognitionLayer(provider="ollama", model="neural-chat")

# Reduce context window
cognition.memory.short_term.max_messages = 5

# Reduce token limits
llm_response = cognition.llm.complete(messages, max_tokens=256)
```

### Cost Optimization (if using cloud)

```python
# Use local Ollama for routine decisions
fast_cognition = CognitionLayer(provider="ollama", model="llama2")

# Use cloud for critical planning only
slow_cognition = CognitionLayer(provider="openai", model="gpt-4")

# Route intelligently
if decision_type == CRITICAL_PLANNING:
    return await slow_cognition.plan_mission(goal)
else:
    return fast_cognition.decide(observation)
```

## Troubleshooting

### Ollama Connection Error
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Check port
netstat -tuln | grep 11434

# Try custom URL
cognition = CognitionLayer(
    provider="ollama",
    config={"base_url": "http://192.168.1.100:11434"}
)
```

### Slow Response Times
```bash
# Use faster model
ollama pull neural-chat  # Smaller, faster

# Check system resources
free -h  # RAM
nvidia-smi  # GPU
```

### Memory Growing Too Large
```python
# Clear short-term context periodically
cognition.memory.short_term.clear()

# Archive old episodic memories
cognition.memory.episodic.missions = cognition.memory.episodic.missions[-100:]
```

## Next Steps

1. **Test locally** with Ollama to understand behavior
2. **Integrate with your fast brain** gradually (start with recovery planning)
3. **Tune prompts** for your specific domain
4. **Monitor performance** and optimize
5. **Deploy to production** with confidence

## Resources

- **Ollama**: https://ollama.ai/
- **Open models**: https://huggingface.co/models
- **LLM benchmarks**: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
- **Prompt engineering**: https://platform.openai.com/docs/guides/prompt-engineering

## Architecture Diagrams

### Data Flow
```
Sensors (50 Hz)
  ↓
Sensor Aggregator (32D vector)
  ↓
Fast Brain (PyTorch, 20 Hz) ←──────────┐
  ↓                                    │
Decision Router                        │ (continuous
  ├→ Critical/High urgency             │  fast loop)
  │   └→ Fast Brain Action              │
  │                                    │
  └→ Medium/Low urgency               │
      ├→ Check Slow Brain Ready       │
      └→ Use Slow Brain if Ready ─┐   │
                                   ↓   ↓
Slow Brain (LLM, 1 Hz, async)    ←────┘
  ├─ Planning Engine
  ├─ Memory Stack
  ├─ Language Interface
  └─ Recovery Strategies
       ↓
   Explanation Publisher
   Plan Publisher
```

### Time Domains
```
Fast Brain (Real-time)
├─ 0-50ms: Sensor processing
├─ 50-100ms: Neural network inference
└─ 100-50ms: Command output
└─ Total: 50-100ms per cycle (20 Hz)

Slow Brain (Planning)
├─ 0-500ms: Context preparation
├─ 500-2000ms: LLM inference
├─ 2000-5000ms: Plan generation
└─ Total: 0.5-5 seconds per cycle (1 Hz)

Both run asynchronously without blocking each other
```

---

**Integration Complete!** Your robot now has both fast reactive control and slow intelligent planning.

