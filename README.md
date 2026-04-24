# SmolVLA Cyberwave Cloud Node

Cloud inference deployment for [SmolVLA](https://huggingface.co/lerobot/smolvla) (Small Vision-Language-Action) models with [Cyberwave](https://cyberwave.com) robot control integration.

This repository provides the infrastructure to run SmolVLA policy inference on Cyberwave Cloud Nodes, enabling real-time robot control from language instructions and camera observations.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Cyberwave Cloud Node                            │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                        deploy.py                              │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    │  │
│  │  │   SmolVLA   │───▶│  predict_fn │───▶│  Action Chunk   │    │  │
│  │  │   Policy    │    │             │    │  (50 actions)   │    │  │
│  │  └─────────────┘    └─────────────┘    └─────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌───────────────────────────▼───────────────────────────────────┐  │
│  │                      CwProcessor                              │  │
│  │  • Cyberwave SDK client                                       │  │
│  │  • MQTT subscription (joint states)                           │  │
│  │  • Camera frame fetching (via twins)                          │  │
│  │  • Action publishing (MQTT)                                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Robot + Cameras │
                    │  (via Cyberwave) │
                    └──────────────────┘
```

## How It Works

1. **Model Loading**: SmolVLA policy is loaded once from a local checkpoint
2. **Observation Collection**: `CwProcessor` fetches camera frames and joint states via Cyberwave SDK
3. **Inference**: The model predicts a chunk of 50 actions from images + state + instruction
4. **Execution**: Actions are published to the robot via MQTT
5. **Loop**: Process repeats for `max_steps` iterations

### Camera Mapping

Training configs contain non-semantic camera names (e.g., `cam_7e7bf9fe`). At runtime, cameras are mapped by position:

```
Training config:  ["cam_7e7bf9fe", "cam_9fcace87", "cam_a6f944f4"]
                       ↓               ↓               ↓
Runtime cameras:  ["primary_camera", "wrist_camera", "overhead_camera"]
```

## Installation

```bash
# Create virtual environment
python3 -m venv ~/.venv/smolvla
source ~/.venv/smolvla/bin/activate

# Install dependencies
./install.sh
```

Or manually:

```bash
pip install -e ".[smolvla]"
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SMOLVLA_CHECKPOINT` | Yes | Path to SmolVLA checkpoint directory |
| `CYBERWAVE_API_KEY` | Yes | Cyberwave API key (set by Cloud Node) |
| `CYBERWAVE_MQTT_HOST` | Yes | MQTT broker host (set by Cloud Node) |
| `CYBERWAVE_MQTT_PORT` | Yes | MQTT broker port (set by Cloud Node) |

### JSON Payload Structure

The Cloud Node sends a JSON payload with robot and camera configuration:

```json
{
  "robot_twin_uuid": "uuid-of-robot-twin",
  "instruction": "pick up the red block and place it in the bin",
  
  "camera_endpoints_by_role": {
    "primary_camera": "camera-twin-uuid-1",
    "wrist_camera": "camera-twin-uuid-2",
    "overhead_camera": "camera-twin-uuid-3"
  },
  
  "twin_calibration": {
    "shoulder_pan": {"min": -3.14, "max": 3.14},
    "shoulder_lift": {"min": -1.57, "max": 1.57},
    "elbow": {"min": -3.14, "max": 3.14},
    "wrist_1": {"min": -3.14, "max": 3.14},
    "wrist_2": {"min": -3.14, "max": 3.14},
    "gripper": {"min": 0, "max": 1.0}
  },
  
  "calibration_robot_type": "follower",
  "max_steps": 100,
  "mode": "live",
  "wait_for_joint_update_seconds": 1.0,
  
  "action_sleep_seconds": 0.1,
  "inference_loop": true
}
```

### Control Loop Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 1 | Maximum total actions to execute |
| `action_sleep_seconds` | 0.1 | Sleep time between publishing each action |
| `inference_loop` | true | If true, run multiple inference cycles; if false, single chunk |
| `wait_for_joint_update_seconds` | 1.0 | Timeout waiting for initial joint state via MQTT |

### Checkpoint Structure

The checkpoint directory should contain:

```
checkpoint/
├── pretrained_model/
│   ├── train_config.json    # Training configuration (camera names, etc.)
│   ├── config.json          # Model configuration
│   └── model.safetensors    # Model weights
└── ...
```

The `train_config.json` is used to extract camera names for proper mapping.

## Usage

### Local Testing

```bash
export SMOLVLA_CHECKPOINT=/path/to/checkpoint
export CYBERWAVE_API_KEY=your-api-key
export CYBERWAVE_MQTT_HOST=mqtt.cyberwave.com
export CYBERWAVE_MQTT_PORT=8883

python deploy.py /path/to/params.json
```

### Cyberwave Cloud Node

The `cyberwave.yml` configures the Cloud Node:

```yaml
cyberwave-cloud-node:
  install_script: ./install.sh
  
  inference: |
    source "$HOME/.venv/smolvla/bin/activate" && \
    python deploy.py {body}
  
  profile_slug: smolvla
```

The `{body}` placeholder is replaced with the path to the JSON params file.

## Project Structure

```
cyberwave-compute-smolvla/
├── deploy.py           # Main entry point - model loading + inference
├── cw_processor.py     # Cyberwave I/O handler (SDK, MQTT, cameras)
├── cyberwave.yml       # Cloud Node configuration
├── install.sh          # Installation script
└── README.md
```

### Key Components

#### `deploy.py`

- Loads SmolVLA policy from checkpoint
- Extracts camera names from `train_config.json`
- Builds `predict_fn` that handles frame remapping and inference
- Orchestrates the full pipeline

#### `cw_processor.py`

- `CwProcessor`: Handles all Cyberwave SDK interactions
  - Creates SDK client with auto-configured MQTT
  - Subscribes to joint state updates
  - Fetches camera frames from twins
  - Publishes action predictions via MQTT
- `InferenceRequest`: Dataclass for request parameters
- `parse_request_payload()`: JSON parsing with snake_case/camelCase support

## Output

The inference returns a JSON result:

```json
{
  "status": "ok",
  "robot_twin_uuid": "...",
  "instruction": "pick up the red block",
  "joint_state_source": "mqtt_subscription",
  "frames": [
    {"key": "primary_camera", "bytes": 123456},
    {"key": "wrist_camera", "bytes": 98765}
  ],
  "current_joints": {"shoulder_pan": 0.1, ...},
  "target_joints": {"shoulder_pan": 0.5, ...},
  "training_cameras": ["cam_7e7bf9fe", "cam_9fcace87"],
  "camera_mapping": {"cam_7e7bf9fe": "primary_camera", ...},
  "steps_executed": 50
}
```

## Requirements

- Python >= 3.10
- CUDA-capable GPU (recommended)
- LeRobot with SmolVLA support
- Cyberwave SDK >= 0.3.46

## Related Projects

- [LeRobot](https://github.com/huggingface/lerobot) - Robot learning framework
- [SmolVLA](https://huggingface.co/lerobot/smolvla) - Small Vision-Language-Action model
- [Cyberwave](https://cyberwave.com) - Robot cloud infrastructure

## License

See LICENSE file for details.
