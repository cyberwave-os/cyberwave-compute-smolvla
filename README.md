# SmolVLA Cyberwave Cloud Node

Cloud inference and training for [SmolVLA](https://huggingface.co/lerobot/smolvla) (Small Vision-Language-Action) models with [Cyberwave](https://cyberwave.com) robot control integration.

This cloud node runs SmolVLA **inference** (`deploy.py` + `CwProcessor`) and optional **training** (`train.py` + `CwTrainer`) on Cyberwave infrastructure, enabling real-time robot control from language instructions and camera observations.

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
│  │  • Cyberwave SDK client (auto-configured)                     │  │
│  │  • Background camera fetchers (daemon threads)                │  │
│  │  • MQTT subscription (joint states)                           │  │
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

1. **Weights Download**: Model weights are fetched from Cyberwave MLModel API (signed URLs) and cached locally
2. **Model Loading**: SmolVLA policy is loaded from the downloaded checkpoint
3. **Camera Binding**: Background daemon threads continuously fetch frames from each camera twin
4. **Observation Collection**: `CwProcessor` reads cached frames and joint states
5. **Inference**: The model predicts a chunk of 50 actions from images + state + instruction
6. **Execution**: Actions are published to the robot via MQTT
7. **Loop**: Process repeats for `max_steps` iterations

### Background Camera Fetching

Each camera twin has a dedicated daemon thread that continuously polls for frames:

```
┌─────────────────────────────────────────────────────────────┐
│  Camera Threads (Background)                                │
│                                                             │
│  camera_wrist thread ──► GET /twins/{uuid}/latest-frame     │
│       │                        │                            │
│       └──────────────────────▶ cache (np.ndarray + bytes)   │
│                                                             │
│  camera_front thread ──► GET /twins/{uuid}/latest-frame     │
│       │                        │                            │
│       └──────────────────────▶ cache (np.ndarray + bytes)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Control Loop                                               │
│                                                             │
│  get_inputs() ──► reads cached frames (instant, no I/O)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

This decouples frame fetching from the inference loop, ensuring consistent frame rates.

### Camera Mapping

Training configs contain camera names (e.g., `camera_wrist`, `camera_front`). At runtime, cameras are mapped from `camera_endpoints_by_role`:

```
camera_endpoints_by_role: {
    "camera_wrist": "https://api.cyberwave.com/api/v1/twins/{uuid}/latest-frame",
    "camera_front": "https://api.cyberwave.com/api/v1/twins/{uuid}/latest-frame"
}
         ↓
Extract UUIDs from URLs
         ↓
Fetch Twin objects via Cyberwave SDK
         ↓
Start background fetcher threads
```

## Installation

```bash
# Create virtual environment
python3 -m venv ~/.venv/smolvla
source ~/.venv/smolvla/bin/activate

# Install dependencies
./install.sh
```

The install script uses `requirements.txt` which includes:
- `numpy`, `Pillow`, `requests` - Core dependencies
- `lerobot[smolvla]` - LeRobot with SmolVLA support
- `peft` - LoRA fine-tuning
- `cyberwave>=0.3.46` - Cyberwave SDK
- `zstandard` - For `.tar.zst` weight archives

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CYBERWAVE_API_KEY` | Yes | Cyberwave API key (set by Cloud Node) |
| `SMOLVLA_CHECKPOINT` | No | Override checkpoint path (otherwise uses `weights_url`) |

### JSON Payload Structure

The Cloud Node sends a JSON payload with robot and camera configuration:

```json
{
  "robot_twin_uuid": "uuid-of-robot-twin",
  "instruction": "pick up the red block and place it in the bin",

  "weights_url": "https://api.cyberwave.com/api/v1/mlmodels/{uuid}/weights",
  "policy_repo_id": "lerobot/smolvla_base",

  "camera_endpoints_by_role": {
    "camera_wrist": "https://api.cyberwave.com/api/v1/twins/{uuid}/latest-frame",
    "camera_front": "https://api.cyberwave.com/api/v1/twins/{uuid}/latest-frame"
  },

  "twin_calibration": {
    "leader": { "...": "..." },
    "follower": { "...": "..." }
  },
  "calibration_robot_type": "follower",

  "mode": "live",
  "max_steps": 1000,
  "wait_for_joint_update_seconds": 1.0,
  "actions_per_cycle": 25,
  "action_sleep_seconds": 0.1,
  "inference_loop": true
}
```

### Weights Download

The `weights_url` points to the Cyberwave MLModel API endpoint which returns a signed URL:

```
GET /api/v1/mlmodels/{uuid}/weights
    → { "signed_url": "https://storage.../checkpoint.tar.zst", "expires_at": "..." }

GET {signed_url}
    → Download and extract .tar.zst archive
```

Supported archive formats:
- `.tar.zst` (Zstandard compressed tar)
- `.tar.gz` / `.tgz` (Gzip compressed tar)
- `.zip` (ZIP archive)

Weights are cached at `~/.cache/cyberwave/weights/` with automatic directory resolution to find `config.json`.

### Control Loop Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | — | e.g. `live` (passed through from platform; informational) |
| `max_steps` | 1 | Maximum total actions to execute |
| `wait_for_joint_update_seconds` | 1.0 | Timeout waiting for first joint state after MQTT subscribe |
| `actions_per_cycle` | 25 | Actions to execute per inference cycle (from 50-action chunk) |
| `action_sleep_seconds` | 0.1 | Sleep time between publishing each action |
| `inference_loop` | true | If true, run multiple inference cycles; if false, single chunk |
| `camera_poll_interval_seconds` | 0.05 | Background camera polling interval |

## Usage

### Local Testing

```bash
export CYBERWAVE_API_KEY=your-api-key

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

## Project Structure

```
smolvla/
├── deploy.py              # Inference entry point (model load + CwProcessor)
├── train.py               # Training entry point (CwTrainer + lerobot)
├── cw_processor.py        # Inference: SDK, MQTT, cameras, weights download
├── cw_trainer.py          # Training: dataset download, metrics, artifact packaging
├── smolvla_resolver.py    # Model-specific metadata (camera mapping, config parsing)
├── smolvla_trainer.py     # SmolVLA TrainPipelineConfig builder
├── base_resolver.py       # Abstract resolver interface
├── base_trainer.py        # Abstract trainer interface
├── requirements.txt       # Python dependencies
├── install.sh             # Installation script
├── cyberwave.yml          # Cloud Node configuration
├── ARCHITECTURE.md        # Detailed architecture documentation
└── README.md
```

### Key Components

#### `deploy.py`
- Parses request payload and downloads weights via `download_weights()`
- Loads SmolVLA policy from checkpoint
- Builds `predict_fn` for inference
- Creates `CwProcessor` and runs the control loop

#### `cw_processor.py`
- `download_weights()`: Fetches weights from MLModel API, extracts archives, caches locally
- `CwProcessor`: Orchestrates all Cyberwave interactions
  - Creates SDK client with auto-configured MQTT
  - Spawns background camera fetcher threads
  - Subscribes to joint state updates
  - Publishes action predictions via MQTT
- `InferenceRequest`: Dataclass for request parameters
- `CameraBinding`: Holds cached frame data per camera

#### `smolvla_resolver.py`
- Loads `train_config.json` from checkpoint
- Extracts training camera names and dimensions
- Builds camera mapping (training name → runtime key)

## Output

Terminal output shows real-time progress:

```
══════════════════════════════════════════════════
  CYBERWAVE SETUP
══════════════════════════════════════════════════
  API Key: cw_a9ce8...4bea
  ✓ Client created
  ✓ MQTT connected
  ✓ Joints received: [-0.01, 0.01, 0.07, 0.02, -0.00, 0.12]
  ✓ Camera camera_wrist ready
  ✓ Camera camera_front ready
  Started 2 background camera fetchers

══════════════════════════════════════════════════
  SMOLVLA CONTROL LOOP
══════════════════════════════════════════════════
  Max steps:        1000
  Actions/cycle:    25

──────────────────────────────────────────────────
  CYCLE 1  │  0/1000 steps (0%)
──────────────────────────────────────────────────
  Cameras: 2 frames  │  State dim: 6
  Running inference...
  Predicted 50 actions -> executing 25
  [████████████████████] 25/25  ✊ [+0.11, -0.00, +0.19, +0.41, -0.15, +0.10]
  ✓ Executed 25 actions
```

JSON result is written to stdout for orchestration:

```json
{
  "status": "ok",
  "robot_twin_uuid": "...",
  "instruction": "put object in box",
  "steps_executed": 1000,
  "initial_joints": {"_1": -0.01, "_2": 0.01, ...}
}
```

## Requirements

- Python >= 3.10
- CUDA-capable GPU (recommended)
- LeRobot with SmolVLA support
- Cyberwave SDK >= 0.3.46

## Related

- [LeRobot](https://github.com/huggingface/lerobot) - Robot learning framework
- [SmolVLA](https://huggingface.co/lerobot/smolvla) - Small Vision-Language-Action model
- [Cyberwave](https://cyberwave.com) - Physical AI Clound Infrastructure

## License

See LICENSE file for details.
