# SmolVLA Cyberwave Compute - Architecture

This document explains the architecture of the SmolVLA inference and training systems for Cyberwave Cloud.

**Table of Contents**
- [Inference Architecture](#inference-architecture)
- [Training Architecture](#training-architecture)

---

# Inference Architecture

## Overview

The inference system is split into three main components with distinct responsibilities:

```
┌─────────────────────────────────────────────────────────────────┐
│                         deploy.py                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  • Load SmolVLA model (torch, PEFT adapters)                ││
│  │  • Build predict_fn(inputs) -> raw tensor                   ││
│  │  • Minimal: no camera/joint/Cyberwave logic                 ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓ predict_fn, checkpoint            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    cw_processor.py                           ││
│  │  • Cyberwave SDK client + MQTT                              ││
│  │  • Builds resolver from checkpoint + model_slug             ││
│  │  • get_inputs(): camera frames → training names, state      ││
│  │  • _convert_raw_actions(): tensor → [{_1, _2, ..._6}]       ││
│  │  • Control loop, action publishing, gripper verification    ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓ checkpoint                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  smolvla_resolver.py                         ││
│  │  • Load train_config.json                                   ││
│  │  • Extract training camera names                            ││
│  │  • Build camera mapping (training ↔ runtime)                ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

### deploy.py - Model Loading Only

**Purpose**: Entry point script that only handles ML model loading.

| Task | Description |
|------|-------------|
| **Model Loading** | Loads SmolVLA from checkpoint (supports full models & PEFT adapters) |
| **Predict Function** | Creates `predict_fn` that takes inputs dict and returns raw tensor |
| **Preprocessing** | Builds observation tensors and runs normalization |

```python
# deploy.py main() simplified
checkpoint = os.environ["SMOLVLA_CHECKPOINT"]
predict_fn = build_predict_fn(checkpoint)

processor = CwProcessor(
    request,
    model_slug="smolvla",
    checkpoint=checkpoint,
    predict_fn=predict_fn,
)
processor.setup()
result = processor.run()
```

The `predict_fn` signature:

```python
def predict(inputs: dict[str, Any]) -> torch.Tensor:
    """
    Args:
        inputs: {
            "images": {training_camera_name: np.ndarray HWC uint8},
            "state": np.ndarray of joint positions,
            "instruction": str
        }
    
    Returns:
        Raw action tensor [1, chunk_size, action_dim]
    """
```

### smolvla_resolver.py - Model-Specific Metadata

**Purpose**: Handles SmolVLA-specific configuration and mappings. No torch, no Cyberwave imports.

| Task | Description |
|------|-------------|
| **Config Loading** | Reads `train_config.json` from checkpoint |
| **Camera Extraction** | Extracts camera names used during training |
| **Camera Mapping** | Maps training names to runtime identifiers by position |

```python
class SmolVLAResolver:
    MODEL_SLUG = "smolvla"
    
    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self.training_config = self._load_training_config()
        self.training_camera_names = self._extract_camera_names()
    
    def build_camera_mapping(
        self,
        runtime_cameras: dict[str, str] | list[str],
    ) -> dict[str, str]:
        """Map training camera names to runtime identifiers (positional)."""
```

### cw_processor.py - Cyberwave I/O & Control Loop

**Purpose**: Handles all Cyberwave communication, data transformations, and robot control.

| Task | Description |
|------|-------------|
| **SDK Client** | Creates and configures Cyberwave client with API credentials |
| **Resolver Management** | Builds resolver from checkpoint via `RESOLVER_REGISTRY` |
| **Input Preparation** | `get_inputs()` fetches frames and remaps to training camera names |
| **Action Conversion** | `_convert_raw_actions()` maps tensor to joint dicts `{_1, _2, ...}` |
| **MQTT** | Connects, subscribes to joints, publishes actions |
| **Control Loop** | Orchestrates observe → predict → convert → execute → repeat |
| **Gripper Verification** | Ensures gripper reaches target before proceeding |

---

## Runtime Flow

```
┌────────────────────────────────────────────────────────────────┐
│                        RUNTIME FLOW                             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DEPLOY.PY - STARTUP                                         │
│     ├─ Load SmolVLA model (PEFT if needed)                      │
│     ├─ Create predict_fn                                        │
│     └─ Create CwProcessor(model_slug, checkpoint, predict_fn)   │
│                                                                 │
│  2. CW_PROCESSOR - SETUP                                        │
│     ├─ Build SmolVLAResolver from checkpoint                    │
│     ├─ Connect to MQTT broker                                   │
│     ├─ Subscribe to joint states                                │
│     ├─ Wait for initial joint state (5s timeout)                │
│     ├─ Derive joint_names (hardcoded: _1, _2, ..., _6)          │
│     └─ Build camera_mapping via resolver                        │
│                                                                 │
│  3. CONTROL LOOP (repeat until max_steps)                       │
│     │                                                           │
│     ├─ GET_INPUTS                                               │
│     │   ├─ Fetch camera frames                                  │
│     │   ├─ Remap runtime keys → training camera names           │
│     │   ├─ Decode JPEG → numpy arrays                           │
│     │   ├─ Build state vector from joint positions              │
│     │   └─ Return {"images": {...}, "state": [...], "instruction": ...}  │
│     │                                                           │
│     ├─ PREDICT                                                  │
│     │   └─ raw_actions = predict_fn(inputs)                     │
│     │       → Returns tensor [1, 50, action_dim]                │
│     │                                                           │
│     ├─ CONVERT_RAW_ACTIONS                                      │
│     │   └─ tensor → [{_1: v, _2: v, ..., _6: v}, ...]           │
│     │                                                           │
│     └─ EXECUTE ACTIONS (first 25 of 50)                         │
│         ├─ For each action:                                     │
│         │   ├─ Publish via MQTT                                 │
│         │   ├─ Sleep (0.1s)                                     │
│         │   └─ If gripper closing: verify & retry               │
│         └─ Re-observe and predict again                         │
│                                                                 │
│  4. COMPLETE                                                    │
│     └─ Return summary JSON                                      │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
                    JSON Request
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      deploy.py                               │
│                                                              │
│   1. Parse request payload                                   │
│   2. Load SmolVLA model → predict_fn                         │
│   3. Pass checkpoint + predict_fn to CwProcessor             │
│                                                              │
└────────────────────────┬────────────────────────────────────┘
                         │ (model_slug, checkpoint, predict_fn)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    cw_processor.py                           │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              SmolVLAResolver (built here)            │   │
│   │  • training_camera_names: ["cam_7e7bf9fe", ...]     │   │
│   │  • build_camera_mapping()                            │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                   MQTT Broker                        │   │
│   │  ◄──── Subscribe to joint states ────►              │   │
│   │  ◄──── Publish joint targets ────────►              │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                  Camera Twins                        │   │
│   │  ◄──── GET /twins/{uuid}/latest-frame ────►         │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
│   Control Loop:                                              │
│     get_inputs() ──► predict_fn(inputs) ──► raw_tensor      │
│                                                │             │
│                                                ▼             │
│                                 _convert_raw_actions()       │
│                                                │             │
│                                                ▼             │
│                                      publish via MQTT        │
│                                                │             │
│                                                ▼             │
│                                          Robot moves         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Classes

### InferenceRequest

```python
@dataclass
class InferenceRequest:
    robot_twin_uuid: str           # Robot to control
    instruction: str               # Task instruction
    camera_twin_uuids: list[str]   # Camera UUIDs
    camera_endpoints_by_role: dict # {"wrist_camera": "uuid", ...}
    max_steps: int                 # Total actions to execute
    actions_per_cycle: int         # Actions per inference (default: 25)
    action_sleep_seconds: float    # Delay between actions (default: 0.1s)
    inference_loop: bool           # Re-predict after each cycle (default: True)
```

### CwProcessor

```python
class CwProcessor:
    def __init__(
        self,
        request: InferenceRequest,
        *,
        model_slug: str,       # e.g., "smolvla"
        checkpoint: str,       # path to checkpoint
        predict_fn: PredictFn, # inputs dict -> raw tensor
    ):
        # Builds resolver from RESOLVER_REGISTRY[model_slug](checkpoint)
    
    def setup(self) -> None:
        """Initialize SDK, MQTT, build camera_mapping, derive joint_names."""
    
    def get_inputs(self) -> dict[str, Any]:
        """Fetch frames, remap to training names, build state vector."""
    
    def _convert_raw_actions(self, raw: Any) -> list[dict[str, float]]:
        """Convert tensor to list of {_1, _2, ..., _6} dicts."""
    
    def run(self) -> dict[str, Any]:
        """Execute the full control loop."""
```

### SmolVLAResolver

```python
class SmolVLAResolver:
    MODEL_SLUG = "smolvla"
    
    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self.training_config = self._load_training_config()
        self.training_camera_names = self._extract_camera_names()
    
    def build_camera_mapping(
        self,
        runtime_cameras: dict[str, str] | list[str],
    ) -> dict[str, str]:
        """training_name -> runtime_key (positional match)."""
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SMOLVLA_CHECKPOINT` | Yes | Path to SmolVLA checkpoint directory |
| `SMOLVLA_BASE_MODEL` | No | For PEFT: path to base model |
| `CYBERWAVE_API_KEY` | Yes | API key for Cyberwave authentication |
| `CYBERWAVE_MQTT_HOST` | No | MQTT broker host |
| `CYBERWAVE_MQTT_PORT` | No | MQTT broker port |
| `CYBERWAVE_MQTT_PASSWORD` | No | MQTT password (defaults to API key) |

---

## Gripper Close Verification

When the gripper is closing (target < 0.3), the system verifies the gripper reaches the target position:

```python
# In cw_processor._execute_action_chunk
gripper_key = "_6"              # Joint 6 is the gripper
gripper_close_threshold = 0.3   # Values below = closing
position_match_threshold = 0.01 # Must match within tolerance
max_gripper_retries = 10        # Max retry attempts

if gripper_target < 0.3:
    while actual_gripper != target (within 0.01):
        resend_action()
        wait(0.1s)
```

---

## Extending to Other Models

The architecture supports adding new models (e.g., OpenVLA) by:

1. Creating a new resolver (e.g., `openvla_resolver.py`) implementing the `ModelResolver` protocol
2. Registering it in `RESOLVER_REGISTRY` in `cw_processor.py`
3. Creating a new `deploy_openvla.py` with model-specific loading

```python
# In cw_processor.py
RESOLVER_REGISTRY = {
    "smolvla": SmolVLAResolver,
    # "openvla": OpenVLAResolver,  # future
}
```

---

## Usage Example

```bash
# Set environment
export CYBERWAVE_API_KEY="cw_your_api_key"
export SMOLVLA_CHECKPOINT="/path/to/checkpoint"

# Run inference
python deploy.py test_params.json
```

### test_params.json

```json
{
  "robot_twin_uuid": "b10e8ffa-f58c-49e0-a9c0-f76ffbed0356",
  "instruction": "pick up the red block and place it in the box",
  "camera_endpoints_by_role": {
    "wrist_camera": "d62cf8b3-9533-496a-95a6-34e8188a885a",
    "top_camera": "69938649-de93-4db1-bd7e-c413114525c0",
    "front_camera": "3d1f467f-500c-45fe-a754-537acdb7b464"
  },
  "max_steps": 100,
  "actions_per_cycle": 25,
  "action_sleep_seconds": 0.1,
  "inference_loop": true
}
```

---

## Output

The system produces colorized terminal output showing:

```
══════════════════════════════════════════════════
  CYBERWAVE SETUP
══════════════════════════════════════════════════
  API Key: cw_dffac...2048
  ✓ Client created
  ✓ MQTT connected
  ✓ Joints received: [0.01, -1.54, 1.48, 0.03, 0.01, 0.31]
  Joint names: ['_1', '_2', '_3', '_4', '_5', '_6']
  Camera mapping: 3 cameras
    cam_7e7bf9fe <- wrist_camera
    cam_9fcace87 <- top_camera
    cam_a6f944f4 <- front_camera

══════════════════════════════════════════════════
  SMOLVLA CONTROL LOOP
══════════════════════════════════════════════════
  Max steps:        100
  Actions/cycle:    25
  Joint source:     mqtt_subscription

──────────────────────────────────────────────────
  CYCLE 1  │  0/100 steps (0%)
──────────────────────────────────────────────────
  Cameras: 3 frames  │  State dim: 6
  State: [0.01, -1.54, 1.48, 0.03, 0.01, 0.31]
  Running inference...
  Predicted 50 actions -> executing 25
  [████████████████████] 25/25  ✋ [+0.11, -1.32, +1.35, +0.25, -0.26, +0.18]
  ✓ Executed 25 actions
```

Final JSON result is written to stdout for orchestration systems.

---

# Training Architecture

## Overview

The training system mirrors the inference architecture with a parallel set of components:

```
┌─────────────────────────────────────────────────────────────────┐
│                         train.py                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  • SmolVLA entry point (MODEL_SLUG = "smolvla")             ││
│  │  • Parse JSON params from Cloud Node                        ││
│  │  • Instantiate CwTrainer and run training                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓ params, model_slug                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    cw_trainer.py                             ││
│  │  • Download dataset from Cyberwave API                      ││
│  │  • Download base model weights (optional)                   ││
│  │  • Monkey-patch WandBLogger → CyberwaveLogger               ││
│  │  • Build TrainPipelineConfig via trainer registry           ││
│  │  • Execute lerobot train(), send status/metrics/ETA         ││
│  │  • Compress results to results_folder                       ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓ params                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  smolvla_trainer.py                          ││
│  │  • Build TrainPipelineConfig for SmolVLA                    ││
│  │  • Configure PEFT/LoRA settings                             ││
│  │  • Set learning rate, batch size, steps                     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Training Components

### train.py - Entry Point

**Purpose**: Minimal SmolVLA-specific entry point for the Cloud Node.

| Task | Description |
|------|-------------|
| **Parse Params** | Load JSON from Cloud Node params file |
| **Create Trainer** | Instantiate `CwTrainer` with `model_slug="smolvla"` |
| **Run Training** | Call `trainer.setup()` then `trainer.run()` |

```python
# train.py simplified
MODEL_SLUG = "smolvla"

def main():
    params = load_json_argument(sys.argv[1])
    trainer = CwTrainer(params, model_slug=MODEL_SLUG)
    trainer.setup()
    result = trainer.run()
    print(json.dumps(result))
```

### cw_trainer.py - Training Orchestrator

**Purpose**: Handles all Cyberwave-specific training concerns.

| Task | Description |
|------|-------------|
| **Dataset Download** | Fetch and extract dataset from `/api/v1/datasets/{uuid}/zip` |
| **Weights Download** | Download custom base weights tar (optional) |
| **Trainer Registry** | Look up model-specific trainer by `model_slug` |
| **Logger Patch** | Replace `WandBLogger` with `CyberwaveLogger` |
| **Config Build** | Delegate to `SmolVLATrainer.build_pipeline_config()` |
| **Training Execution** | Call `lerobot.scripts.lerobot_train.train(cfg)` |
| **Status Updates** | POST metrics/ETA to `/api/v1/mltrainings/{uuid}` |
| **Artifact Compression** | Tar the final checkpoint to `results_folder` |

```python
class CwTrainer:
    def __init__(self, params: dict, *, model_slug: str):
        # Look up trainer in registry
        registry = _get_trainer_registry()
        self.trainer = registry[model_slug]()
    
    def setup(self) -> None:
        self._download_dataset()
        self._resolve_base_model()
        self._build_training_config()
        self._patch_wandb_logger()
    
    def run(self) -> dict:
        train(self.training_cfg)
        artifact_path = self._compress_results()
        self._cyberwave_logger.mark_completed(str(artifact_path))
        return {"status": "completed", "artifact": str(artifact_path)}
```

### smolvla_trainer.py - Config Builder

**Purpose**: SmolVLA-specific training configuration.

| Task | Description |
|------|-------------|
| **Policy Config** | Load PreTrainedConfig from base model |
| **PEFT Setup** | Configure LoRA with rank from params |
| **Training Settings** | Set steps, batch size, LR, save frequency |

```python
class SmolVLATrainer(BaseVLATrainer):
    MODEL_SLUG = "smolvla"
    
    def build_pipeline_config(self, params, *, dataset_root, ...):
        policy = PreTrainedConfig.from_pretrained(base_model_path)
        policy.device = "cuda"
        policy.optimizer_lr = params.get("optimizer_lr", 1e-3)
        
        peft = PeftConfig(method_type="LORA", r=params.get("lora_r", 16))
        
        return TrainPipelineConfig(
            policy=policy,
            dataset=DatasetConfig(repo_id=dataset_repo_id, root=str(dataset_root)),
            peft=peft,
            steps=params.get("max_steps", 50000),
            wandb=WandBConfig(enable=True, disable_artifact=True),
            ...
        )
```

### CyberwaveLogger - Metrics Reporting

**Purpose**: Drop-in replacement for lerobot's `WandBLogger` that sends metrics to Cyberwave.

| Method | Description |
|--------|-------------|
| `log_dict(d, step, mode)` | POST metrics to `/api/v1/mltrainings/{uuid}` with `update_type="metrics"` |
| `log_policy(checkpoint_dir)` | Log checkpoint save (no-op, cloud node handles uploads) |
| `log_video(...)` | No-op |
| `mark_completed(path)` | POST completion status |

The logger also computes and sends ETA after ~100 steps via `update_type="estimate"`.

---

## Training Data Flow

```
                    JSON Params
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      train.py                                │
│                                                              │
│   1. Parse params from Cloud Node                            │
│   2. Create CwTrainer(model_slug="smolvla")                  │
│                                                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    cw_trainer.py                             │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Cyberwave API                           │   │
│   │  ──► GET /datasets/{uuid}/zip (download dataset)    │   │
│   │  ──► GET weights_url (download base model)          │   │
│   │  ◄── PUT /mltrainings/{uuid} (metrics, ETA)         │   │
│   │  ◄── PUT /mltrainings/{uuid} (completion)           │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              SmolVLATrainer (registry lookup)        │   │
│   │  • build_pipeline_config() → TrainPipelineConfig    │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              CyberwaveLogger (monkey-patched)        │   │
│   │  • Replaces WandBLogger                              │   │
│   │  • log_dict() → PUT /mltrainings/{uuid}             │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              lerobot train(cfg)                      │   │
│   │  • Loads dataset                                     │   │
│   │  • Fine-tunes model with PEFT/LoRA                  │   │
│   │  • Saves checkpoints                                 │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Results Compression                     │   │
│   │  • tar.gz checkpoint → results_folder               │   │
│   │  • Cloud Node uploads to storage                    │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Training API Interactions

The training system communicates with the Cyberwave API at several points:

### 1. Dataset Download

```
GET /api/v1/datasets/{dataset_uuid}/zip
    → Returns: { "zip_url": "https://signed-url..." }

GET {signed_url}
    → Returns: Dataset ZIP file
```

### 2. Weights Download (Optional)

```
GET {weights_url}
    → If JSON: { "signed_url": "...", "checkpoint_path": "..." }
    → Otherwise: Direct tar download
```

### 3. Metrics Updates

```
PUT /api/v1/mltrainings/{training_uuid}
{
    "metadata": {
        "step": 100,
        "log": {
            "train/Loss": 0.123,
            "train/Learning Rate": 0.001
        }
    },
    "update_type": "metrics"
}
```

### 4. ETA Updates (after ~100 steps)

```
PUT /api/v1/mltrainings/{training_uuid}
{
    "metadata": {
        "estimated_end_time": "2025-01-15T18:30:00Z",
        "estimated_remaining_seconds": 14400,
        "estimated_remaining_human": "4h 0m",
        "max_steps": 50000,
        "seconds_per_step": 0.288
    },
    "update_type": "estimate"
}
```

### 5. Completion

```
PUT /api/v1/mltrainings/{training_uuid}
{
    "status": "completed",
    "metadata": {
        "completion_source": "cw_trainer.py",
        "local_checkpoint_path": "./runs/artifacts/uuid.tar.gz"
    }
}
```

---

## Extending Training to Other Models

The training architecture supports adding new models by:

1. **Create a new trainer** (e.g., `openvla_trainer.py`) implementing `BaseVLATrainer`
2. **Register in trainer registry** in `cw_trainer.py`
3. **Create entry point** (e.g., `train_openvla.py`) with the appropriate `MODEL_SLUG`

```python
# base_trainer.py
class BaseVLATrainer(ABC):
    MODEL_SLUG: str
    
    @abstractmethod
    def build_pipeline_config(self, params, *, dataset_root, ...):
        pass

# openvla_trainer.py (future)
class OpenVLATrainer(BaseVLATrainer):
    MODEL_SLUG = "openvla"
    
    def build_pipeline_config(self, ...):
        # OpenVLA-specific config
        pass

# cw_trainer.py
def _get_trainer_registry():
    from smolvla_trainer import SmolVLATrainer
    # from openvla_trainer import OpenVLATrainer  # future
    return {
        SmolVLATrainer.MODEL_SLUG: SmolVLATrainer,
        # OpenVLATrainer.MODEL_SLUG: OpenVLATrainer,
    }
```

---

## Training Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CYBERWAVE_API_KEY` | Yes | API key for authentication (fallback if not in params) |
| `CYBERWAVE_ENVIRONMENT` | No | "production" or "development" (default: production) |
| `CYBERWAVE_API_URL` | No | Override API base URL |
| `CYBERWAVE_RUNTIME_ROOT` | No | Directory for downloads/cache (default: /data/cyberwave_runtime) |

---

## Training Usage Example

```bash
# Set environment
export CYBERWAVE_API_KEY="cw_your_api_key"
export CYBERWAVE_ENVIRONMENT="production"

# Run training
python train.py /path/to/params.json
```

### params.json

```json
{
  "cyberwave_training_uuid": "abc123-def456",
  "dataset_uuid": "dataset-uuid-here",
  "dataset_name": "my-robot-dataset",
  "data_root_dir": "./datasets",
  "base_model": "lerobot/smolvla_base",
  "max_steps": 50000,
  "batch_size": 32,
  "lora_r": 16,
  "save_freq": 10000,
  "log_freq": 100
}
```

---

## Training Output

The training system produces colorized terminal output:

```
============================================================
  SmolVLA Training - Cyberwave Cloud Node
============================================================

Received 12 parameters

=== CwTrainer Initialized ===
  Model: smolvla
  Training UUID: abc123-def456
  Environment: production

--- Setup Phase ---
  Downloading dataset: my-robot-dataset (dataset-uuid-here)
    Progress: 100.0%
  Dataset extracted to ./datasets/my-robot-dataset
  Using base model: lerobot/smolvla_base
  Building training config...
  Config built: 50000 steps, batch_size=32
  Patching WandBLogger -> CyberwaveLogger...
  Patched lerobot.rl.wandb_utils.WandBLogger
  Set WANDB_MODE=disabled
  Setup complete

--- Training Phase ---
  Status: training
  Starting lerobot training...
  [Cyberwave] Logged 5 metrics at step 100
  [Cyberwave] Updated ETA: 2025-01-15T18:30:00Z (~4h 0m remaining)
  ...

--- Compression Phase ---
  Status: compressing
  Compressing ./outputs/train/abc123/checkpoints/last -> ./runs/artifacts/abc123-def456.tar.gz
  Artifact created: ./runs/artifacts/abc123-def456.tar.gz (1234.5MB)

--- Completion ---
  Completion status sent to Cyberwave

============================================================
  Training Complete
============================================================

{"status": "completed", "artifact": "./runs/artifacts/abc123-def456.tar.gz", "training_uuid": "abc123-def456"}
```
