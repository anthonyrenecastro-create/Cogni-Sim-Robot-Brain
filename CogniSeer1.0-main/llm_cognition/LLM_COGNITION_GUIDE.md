# LLM Cognition Layer: Universal AI Brain for CogniSeer Robot

## Overview

The cognition layer adds an intelligent slow brain (LLM-based planning and reasoning) alongside the fast brain (PyTorch neural network). This hybrid architecture enables sophisticated mission planning and recovery while maintaining real-time reactive control.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Cognition Layer (Slow Brain)              │
│  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐│
│  │    LLM     │  │ Planning │  │ Language │  │   Memory   ││
│  │ (Ollama,   │  │ Engine   │  │Interface │  │   Stack    ││
│  │OpenAI,etc) │  │          │  │          │  │(4 types)   ││
│  └────────────┘  └──────────┘  └──────────┘  └────────────┘│
│                                                               │
│              Decision Router (Hybrid Strategy)               │
│         Routes decisions: FAST (urgent) vs SLOW (complex)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                   ▼
   ┌─────────┐        ┌──────────┐       ┌──────────┐
   │  Fast   │        │  Sensors │       │ Actuator │
   │  Brain  │        │          │       │ Commands │
   │(PyTorch)│        │(32D obs) │       │(Velocity)│
   └─────────┘        └──────────┘       └──────────┘
   20 Hz Control Loop - Real-time Reactive
```

## Components

### 1. LLM Provider Interface (Universal)

Pluggable LLM support with providers for:
- **Ollama** (local models: Llama2, Mistral, etc.) — Edge-first
- **OpenAI** (GPT-4, GPT-4V with vision)
- **Anthropic Claude** (Claude 3 with vision)
- **HuggingFace** (open-source models via API)
- **LiteLLM** (unified proxy for 100+ models)

**Design Pattern**: Abstract base class `LLMProvider` ensures any provider can be swapped without code changes.

```python
from llm_cognition import CognitionLayer

# Switch providers easily
cognition_local = CognitionLayer(provider="ollama", model="llama2")
cognition_cloud = CognitionLayer(provider="openai", model="gpt-4")
cognition_claude = CognitionLayer(provider="claude", model="claude-3-sonnet")
```

### 2. Memory Stack (4 Types)

#### Short-Term Memory
- Recent conversation context (last N messages)
- Used for current decision-making
- Lightweight, always in context window
- Size: ~10 messages

#### Long-Term Memory
- Semantic knowledge base (embeddings)
- Persistent across sessions
- Enables learning and knowledge retrieval
- Saved to disk as JSON

#### Episodic Memory
- Mission logs and outcomes
- Records: goal, observations, decisions, results, lessons learned
- Enables learning from experience
- Used for recovery strategy selection

#### Spatial Memory
- Environmental map with landmarks and obstacles
- Position-based queries
- Confidence decay over time
- Used for planning around environment

### 3. Decision Router (Hybrid Strategy)

Routes decisions based on **urgency** and **complexity**:

| Urgency | Decision Type | Brain | Frequency | Latency |
|---------|---------------|-------|-----------|---------|
| Critical | Emergency stop, collision | Fast | 20 Hz | <100ms |
| High | Obstacle avoidance | Fast | 20 Hz | <500ms |
| Medium | Velocity control, navigation | Fast/Slow | Variable | <5s |
| Low | Planning, recovery, communication | Slow | 0.1-1 Hz | <30s |

**Fast Brain**: PyTorch neural network, real-time reactive
**Slow Brain**: LLM with reasoning, async planning

### 4. Planning Engine

- Creates plans for achieving goals
- Handles multi-step missions
- Plans recovery strategies for failures
- Generates step-by-step action sequences

### 5. Language Interface

Generates natural language:
- Decision explanations
- Mission summaries
- Status reports
- Error messages with recovery guidance

## Setup

### Installation

```bash
# Core cognition layer
pip install numpy requests

# For Ollama (local, recommended for edge)
pip install requests

# For OpenAI
pip install openai tiktoken

# For Claude
pip install anthropic

# For HuggingFace
pip install huggingface_hub

# For LiteLLM (unified proxy)
pip install litellm
```

### Setting Up Ollama (Edge-First)

```bash
# 1. Install Ollama
curl https://ollama.ai/install.sh | sh

# 2. Pull a model
ollama pull llama2              # 7B, fast, good quality
ollama pull mistral             # Fast, good reasoning
ollama pull neural-chat         # Good for conversation
ollama pull dolphin-mixtral     # Better reasoning

# 3. Start server (runs on localhost:11434)
ollama serve

# 4. Configure CogniSeer
export OLLAMA_MODEL=llama2  # or your choice
```

### Setting Up Cloud Providers

**OpenAI:**
```bash
export OPENAI_API_KEY="your-key-here"
```

**Claude:**
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage Examples

### Basic Usage

```python
from llm_cognition import CognitionLayer
from llm_cognition.decision_router import DecisionType, Urgency

# Initialize (defaults to local Ollama)
cognition = CognitionLayer(provider="ollama", model="llama2")

# Make a routine decision (fast brain)
decision = cognition.decide(
    observation=sensor_data_32d,  # 32D observation vector
    decision_type=DecisionType.OBSTACLE_AVOIDANCE,
    urgency=Urgency.HIGH,
    goal="navigate_to_target"
)

print(f"Action: {decision.action}")
print(f"Confidence: {decision.confidence}")
print(f"Used LLM: {decision.use_llm}")
```

### Async Planning (Slow Brain)

```python
import asyncio

# Plan a complex mission asynchronously
async def robot_loop():
    # Start mission
    mission_id = cognition.start_mission("explore_area_a")
    
    # Async planning doesn't block fast control loop
    plan = await cognition.plan_mission(
        goal="explore_area_a",
        context="Area A has obstacles to avoid"
    )
    
    # ... fast brain runs at 20 Hz in parallel ...
    
    # Handle failures with LLM-based recovery
    if failure_detected:
        recovery = await cognition.handle_failure(
            failure="stuck_on_obstacle",
            context="Attempted multiple directions"
        )
    
    # End mission
    cognition.end_mission(success=True, outcome="Area explored successfully")

asyncio.run(robot_loop())
```

### Memory Usage

```python
# Add observations
cognition.add_observation("Detected obstacle at 0.5m ahead")

# Query memory for context
context = cognition.memory.get_full_context(
    robot_position=(5.0, 2.0),
    context_radius=5.0
)

# Search long-term memory
relevant = cognition.memory.long_term.search(
    query="how to handle obstacles",
    embedding=query_embedding
)

# Get mission history
successful = cognition.memory.episodic.get_successful_missions(
    goal_filter="navigate"
)
```

### Vision Support (VLM)

```python
from llm_cognition.providers.base import Message

# Use vision if provider supports it
if cognition.llm.supports_vision():
    messages = [
        Message(
            role="user",
            content="What obstacles do you see?",
            images=["robot_camera_image.jpg"]
        )
    ]
    
    response = cognition.llm.complete(messages, max_tokens=256)
    print(response.content)
```

## Configuration

### Default Configuration

```python
config = {
    "base_url": "http://localhost:11434",  # Ollama server
    "temperature": 0.7,  # Lower for factual, higher for creative
    "timeout": 120,  # Request timeout
    "max_tokens": 1024,  # Max response length
}

cognition = CognitionLayer(
    provider="ollama",
    model="llama2",
    config=config,
    memory_persistence_dir="./robot_memory"
)
```

### Provider-Specific Options

**Ollama:**
```python
config = {
    "base_url": "http://192.168.1.100:11434",  # Custom server
    "temperature": 0.5,
}
```

**OpenAI:**
```python
config = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4",  # or gpt-4-vision
}
```

**Claude:**
```python
config = {
    "api_key": os.getenv("ANTHROPIC_API_KEY"),
}
```

## Integration with ROS2

### ROS2 Node Integration

```python
import rclpy
from rclpy.node import Node
from llm_cognition import CognitionLayer
from std_msgs.msg import String
from nav_msgs.msg import Odometry

class CognitionNode(Node):
    def __init__(self):
        super().__init__('cognition_node')
        
        # Initialize cognition layer
        self.cognition = CognitionLayer(provider="ollama", model="llama2")
        
        # Subscribers
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Publishers
        self.explanation_pub = self.create_publisher(String, '/brain/explanation', 10)
        
        # Timer for slow brain (1 Hz)
        self.create_timer(1.0, self.slow_brain_callback)
    
    def odom_callback(self, msg):
        # Fast brain decision
        decision = self.cognition.decide(
            observation=extract_features(msg),
            decision_type=DecisionType.NAVIGATION,
            urgency=Urgency.MEDIUM,
        )
    
    def slow_brain_callback(self):
        # Async slow brain planning
        asyncio.create_task(
            self.cognition.plan_mission("navigate_to_goal")
        )
```

## Decision Making Examples

### Fast Brain (Reactive)
- Obstacle detected → Turn right (fast, heuristic)
- Collision imminent → Emergency stop (20Hz, <50ms)
- Clear path → Continue (immediate)

### Slow Brain (Planning)
- "Plan mission to explore building" → LLM creates step-by-step plan
- "Stuck on obstacle" → LLM suggests recovery with reasoning
- User asks "Why did you stop?" → LLM explains decision with context

## Performance

### Latency Budgets

| Operation | Latency | Where |
|-----------|---------|-------|
| Fast brain decision | <50ms | Real-time |
| Sensor data fusion | <50ms | Aggregator |
| TF broadcasting | <50ms | Transform layer |
| Slow brain decision | <5s | Async, non-blocking |
| LLM inference (Ollama 7B) | 1-3s | CPU/GPU |

### Resource Requirements

| Component | Local (Ollama) | Cloud (OpenAI) |
|-----------|-----------------|-----------------|
| Memory | 4-8GB (7B model) | Minimal |
| CPU | 2+ cores | Network only |
| Network | None required | Required |
| Cost | $0 | Pay-per-token |
| Privacy | 100% local | Uploaded to cloud |

## Best Practices

### 1. Provider Selection

- **Edge deployment**: Use Ollama with Llama2/Mistral (local, free)
- **Best quality**: Use GPT-4 or Claude 3 (cloud, paid)
- **Hybrid**: Use LiteLLM to fallback: Ollama → OpenAI if needed

### 2. Memory Management

- Limit short-term context to recent messages only
- Periodically archive old episodic memories
- Use spatial memory decay for dynamic environments

### 3. Urgency Handling

```python
# Critical decisions: Always fast brain
if urgency == Urgency.CRITICAL:
    use_fast_brain_only()

# Complex decisions: Async slow brain, don't block
if urgency == Urgency.LOW:
    asyncio.create_task(plan_with_llm())
```

### 4. Cost Optimization

- Use local Ollama for routine decisions
- Use cloud LLM (OpenAI/Claude) for critical planning
- Cache LLM responses when possible
- Use temperature=0 for deterministic outputs when needed

## Troubleshooting

### Ollama Connection Failed
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Or use custom URL
cognition = CognitionLayer(
    provider="ollama",
    model="llama2",
    config={"base_url": "http://192.168.1.100:11434"}
)
```

### LLM Responses Too Long
```python
# Limit response length
response = cognition.llm.complete(
    messages,
    max_tokens=256  # Shorter responses
)
```

### Memory Growing Too Large
```python
# Manage memory growth
cognition.memory.short_term.clear()  # Clear old context
cognition.memory.episodic._save_to_disk()  # Archive
```

## Advanced: Custom Providers

Create a custom LLM provider:

```python
from llm_cognition.providers.base import LLMProvider, Message, LLMResponse, ModelInfo

class MyCustomProvider(LLMProvider):
    def _initialize(self):
        pass  # Your init code
    
    def complete(self, messages, temperature=0.7, max_tokens=1024, **kwargs):
        # Your API call
        return LLMResponse(
            content="response",
            model=self.model_name,
            tokens_used={"input": 100, "output": 50},
            stop_reason="stop"
        )
    
    # Implement other abstract methods...
```

## References

- **Ollama**: https://ollama.ai/
- **OpenAI API**: https://platform.openai.com/
- **Claude API**: https://console.anthropic.com/
- **HuggingFace**: https://huggingface.co/
- **LiteLLM**: https://github.com/BerriAI/litellm
- **Local Models**: https://huggingface.co/models?sort=trending

---

**Next Steps:**
1. Choose your LLM provider (start with Ollama for edge deployment)
2. Configure and test with your robot
3. Fine-tune prompts for your domain
4. Integrate with ROS2 nodes
5. Monitor and optimize performance

