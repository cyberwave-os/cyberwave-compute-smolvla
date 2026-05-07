# SmolVLA Cyberwave Cloud Node - Architecture

This document explains the architecture of the SmolVLA inference and training systems for [Cyberwave](https://cyberwave.com) Cloud.

**Table of Contents**
- [Inference Architecture](#inference-architecture)
- [Training Architecture](#training-architecture)

---

# Inference Architecture

## Overview

The inference system is split into three main components with distinct responsibilities:

```
┌─────────────────────────────────────────────────────────────────┐
│                         deploy.py                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  • Parse request, download weights from Cyberwave API       ││
│  │  • Load SmolVLA model (torch, PEFT adapters)                ││
│  │  • Build predict_fn(inputs) -> raw tensor                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓ predict_fn, checkpoint           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    cw_processor.py                          ││
│  │  • Cyberwave SDK client (auto-configured)                   ││
│  │  • Background camera daemon threads                         ││
│  │  • MQTT joint subscription + action publishing              ││
│  │  • get_inputs(): cached frames + state                      ││
│  │  • _convert_raw_actions(): tensor → [{_1, _2, ..._6}]       ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓ checkpoint                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  smolvla_resolver.py                        ││
│  │  • Load train_config.json                                   ││
│  │  • Extract training camera names + dimensions               ││
│  │  • Build camera mapping (training ↔ runtime)                ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

### deploy.py - Entry Point & Model Loading

**Purpose**: Entry point that handles weights download and ML model loading.

| Task | Description |
|------|-------------|
| **Weights Download** | Fetches weights from `weights_url` via `download_weights()` |
| **Model Loading** | Loads SmolVLA from checkpoint (supports full models & PEFT adapters) |
| **Predict Function** | Creates `predict_fn` that takes inputs dict and returns raw tensor |

```python
# deploy.py main() simplified
request = parse_request_payload(sys.argv[1])

# Download weights from Cyberwave MLModel API
if request.weights_url:
    checkpoint = download_weights(request.weights_url)
else:
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

### smolvla_resolver.py - Model-Specific Metadata

**Purpose**: Handles SmolVLA-specific configuration and mappings. No torch, no Cyberwave imports.

| Task | Description |
|------|-------------|
| **Config Loading** | Reads `train_config.json` from checkpoint |
| **Camera Extraction** | Extracts camera names used during training |
| **Dimension Extraction** | Extracts expected state/action dimensions |
| **Camera Mapping** | Maps training names to runtime identifiers |

```python
class SmolVLAResolver:
    MODEL_SLUG = "smolvla"
    
    def __init__(self, checkpoint: str) -> None:
        self.training_config = self._load_training_config()
        self.training_camera_names = self._extract_camera_names()
        self.expected_state_dim = self._extract_state_dim()
        self.expected_action_dim = self._extract_action_dim()
    
    def build_camera_mapping(
        self,
        runtime_cameras: dict[str, str] | list[str],
    ) -> dict[str, str]:
        """Map training camera names to runtime identifiers."""
```

### cw_processor.py - Cyberwave I/O & Control Loop

**Purpose**: Handles all Cyberwave communication, data transformations, and robot control.

| Task | Description |
|------|-------------|
| **Weights Download** | `download_weights()` fetches from MLModel API, extracts archives |
| **SDK Client** | Creates Cyberwave client with auto-configured MQTT |
| **Camera Bindings** | Creates `CameraBinding` per camera with background fetcher threads |
| **Input Preparation** | `get_inputs()` reads cached frames (no I/O during inference) |
| **Action Conversion** | `_convert_raw_actions()` maps tensor to joint dicts |
| **MQTT** | Subscribes to joints, publishes actions |
| **Gripper Verification** | Ensures gripper reaches target before proceeding |

---

## Weights Download Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     download_weights()                          │
│                                                                 │
│   1. GET /api/v1/mlmodels/{uuid}/weights                        │
│      ├─ Response: { "signed_url": "...", "expires_at": "..." }  │
│      └─ Extract signed_url from JSON                            │
│                                                                 │
│   2. GET {signed_url}                                           │
│      └─ Stream download to temp file                            │
│                                                                 │
│   3. Detect archive type (magic bytes + headers)                │
│      ├─ .tar.zst (0x28B52FFD) → zstandard + tarfile             │
│      ├─ .tar.gz  (0x1F8B)     → tarfile                         │
│      └─ .zip     (PK)         → zipfile                         │
│                                                                 │
│   4. Extract to ~/.cache/cyberwave/weights/{hash}/              │
│                                                                 │
│   5. Resolve model directory (find config.json)                 │
│      ├─ Check base dir                                          │
│      ├─ Check pretrained_model/                                 │
│      └─ Recurse into single subdirectories (max depth 3)        │
│                                                                 │
│   6. Return resolved checkpoint path                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Background Camera Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    _setup_camera_bindings()                     │
│                                                                 │
│   For each camera in camera_mapping:                            │
│   1. Extract UUID from endpoint URL                             │
│   2. Fetch Twin object via SDK                                  │
│   3. Create CameraBinding(role, twin_uuid, twin)                │
│   4. Spawn daemon thread running _camera_loop()                 │
│                                                                 │
│   Wait for all cameras to fetch first frame (10s timeout)       │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      _camera_loop()                             │
│                     (per-camera daemon thread)                  │
│                                                                 │
│   while not stop_event:                                         │
│       raw = twin.get_latest_frame()                             │
│       if len(raw) >= min_valid_bytes:                           │
│           img = decode_jpeg(raw)  # PIL → numpy                 │
│           with lock:                                            │
│               binding.latest_bytes = raw                        │
│               binding.latest_image = img                        │
│               binding.last_ts = now()                           │
│       sleep(poll_interval)  # default 50ms                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       get_inputs()                              │
│                    (called by control loop)                     │
│                                                                 │
│   with lock:                                                    │
│       for role, binding in cameras.items():                     │
│           images[role] = binding.latest_image  # instant read   │
│                                                                 │
│   state = [joints[name] for name in joint_names]                │
│   return {"images": images, "state": state, "instruction": ...} │
└─────────────────────────────────────────────────────────────────┘
```

**Benefits**:
- Frame fetching is decoupled from inference loop
- No I/O latency during `get_inputs()` - just reads cached numpy arrays
- Consistent frame timing regardless of inference speed
- TODO: Replace REST polling with WebRTC once available

---

## Runtime Flow

```
┌────────────────────────────────────────────────────────────────┐
│                        RUNTIME FLOW                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. DEPLOY.PY - STARTUP                                        │
│     ├─ Parse request payload                                   │
│     ├─ Download weights from Cyberwave MLModel API             │
│     ├─ Load SmolVLA model (PEFT if needed)                     │
│     ├─ Create predict_fn                                       │
│     └─ Create CwProcessor(model_slug, checkpoint, predict_fn)  │
│                                                                │
│  2. CW_PROCESSOR - SETUP                                       │
│     ├─ Create Cyberwave SDK client (auto-configured)           │
│     ├─ Connect to MQTT broker                                  │
│     ├─ Subscribe to joint states                               │
│     ├─ Wait for initial joint state (5s timeout)               │
│     ├─ Derive joint_names from twin schema                     │
│     ├─ Build camera_mapping via resolver                       │
│     ├─ Create CameraBinding + start background threads         │
│     └─ Wait for all cameras ready (10s timeout)                │
│                                                                │
│  3. CONTROL LOOP (repeat until max_steps)                      │
│     │                                                          │
│     ├─ GET_INPUTS                                              │
│     │   ├─ Read cached camera frames (instant, no I/O)         │
│     │   ├─ Build state vector from joint positions             │
│     │   └─ Return {"images": {...}, "state": [...], ...}       │
│     │                                                          │
│     ├─ PREDICT                                                 │
│     │   └─ raw_actions = predict_fn(inputs)                    │
│     │       → Returns tensor [1, 50, action_dim]               │
│     │                                                          │
│     ├─ CONVERT_RAW_ACTIONS                                     │
│     │   └─ tensor → [{_1: v, _2: v, ..., _6: v}, ...]          │
│     │                                                          │
│     └─ EXECUTE ACTIONS (first 25 of 50)                        │
│         ├─ For each action:                                    │
│         │   ├─ Publish via MQTT                                │
│         │   ├─ Sleep (0.1s)                                    │
│         │   └─ If gripper closing: verify & retry              │
│         └─ Re-observe and predict again                        │
│                                                                │
│  4. DISCONNECT                                                 │
│     ├─ Signal camera threads to stop                           │
│     ├─ Join all camera threads                                 │
│     └─ Disconnect MQTT client                                  │
│                                                                │
│  5. RETURN RESULT                                              │
│     └─ JSON with status, steps_executed, initial_joints        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
                    JSON Request
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      deploy.py                              │
│                                                             │
│   1. Parse request payload                                  │
│   2. Download weights from weights_url                      │
│   3. Load SmolVLA model → predict_fn                        │
│   4. Pass checkpoint + predict_fn to CwProcessor            │
│                                                             │
└────────────────────────┬────────────────────────────────────┘
                         │ (model_slug, checkpoint, predict_fn)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    cw_processor.py                          │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              SmolVLAResolver (built here)           │   │
│   │  • training_camera_names: ["camera_wrist", ...]     │   │
│   │  • expected_state_dim: 6                            │   │
│   │  • build_camera_mapping()                           │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │           Background Camera Threads                 │   │
│   │  ◄──── GET /twins/{uuid}/latest-frame ────►         │   │
│   │  ◄──── Cache decoded np.ndarray + bytes ────►       │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                   MQTT Broker                       │   │
│   │  ◄──── Subscribe to joint states ────►              │   │
│   │  ◄──── Publish joint targets ────────►              │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   Control Loop:                                             │
│     get_inputs() ──► predict_fn(inputs) ──► raw_tensor      │
│                                                │            │
│                                                ▼            │
│                                 _convert_raw_actions()      │
│                                                │            │
│                                                ▼            │
│                                      publish via MQTT       │
│                                                │            │
│                                                ▼            │
│                                          Robot moves        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Classes

### CameraBinding

```python
@dataclass
class CameraBinding:
    """Holds state for a camera twin used by the inference loop."""
    role: str                           # training_name (e.g. "camera_wrist")
    twin_uuid: str                      # Extracted from endpoint URL
    twin: Any                           # SDK Twin handle
    latest_bytes: bytes = b""           # Raw JPEG bytes
    latest_image: np.ndarray | None     # Decoded HWC uint8 RGB
    last_ts: float = 0.0                # Timestamp of last successful fetch
```

### InferenceRequest

```python
@dataclass
class InferenceRequest:
    robot_twin_uuid: str                    # Robot to control
    instruction: str                        # Task instruction
    camera_endpoints_by_role: dict[str, str]  # {"camera_wrist": "https://.../latest-frame"}
    weights_url: str | None                 # MLModel weights API endpoint
    policy_repo_id: str | None              # HuggingFace repo (fallback)
    max_steps: int                          # Total actions to execute
    actions_per_cycle: int                  # Actions per inference (default: 25)
    action_sleep_seconds: float             # Delay between actions (default: 0.1s)
    camera_poll_interval_seconds: float     # Background fetch interval (default: 0.05s)
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
        cw: Cyberwave | None,  # optional injected client (for testing)
    ):
        # Builds resolver from RESOLVER_REGISTRY[model_slug](checkpoint)
        self.cameras: dict[str, CameraBinding] = {}
        self._camera_threads: list[threading.Thread] = []
    
    def setup(self) -> None:
        """Initialize SDK, MQTT, cameras, joint_names."""
    
    def get_inputs(self) -> dict[str, Any]:
        """Read cached frames and build inputs dict."""
    
    def _convert_raw_actions(self, raw: Any) -> list[dict[str, float]]:
        """Convert tensor to list of {_1, _2, ..., _6} dicts."""
    
    def run(self) -> dict[str, Any]:
        """Execute the full control loop."""
    
    def disconnect(self) -> None:
        """Stop camera threads and disconnect MQTT."""
```

### SmolVLAResolver

```python
class SmolVLAResolver:
    MODEL_SLUG = "smolvla"
    
    def __init__(self, checkpoint: str) -> None:
        self.training_config = self._load_training_config()
        self.training_camera_names = self._extract_camera_names()
        self.expected_state_dim = self._extract_state_dim()
        self.expected_action_dim = self._extract_action_dim()
    
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
| `CYBERWAVE_API_KEY` | Yes | API key for Cyberwave authentication |
| `SMOLVLA_CHECKPOINT` | No | Override checkpoint path (otherwise uses weights_url) |
| `CYBERWAVE_MQTT_HOST` | No | MQTT broker host (auto-configured by SDK) |
| `CYBERWAVE_MQTT_PORT` | No | MQTT broker port (auto-configured by SDK) |

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

1. Creating a new resolver (e.g., `openvla_resolver.py`) implementing the `BaseVLAResolver` protocol
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

# Run inference (weights downloaded from weights_url in params)
python deploy.py test_params.json
```

### test_params.json

```json
{
  "robot_twin_uuid": "e305bb3e-8c5f-4bf7-807b-21cdb24c88fc",
  "instruction": "put object in box",
  "weights_url": "https://api.cyberwave.com/api/v1/mlmodels/{uuid}/weights",
  "policy_repo_id": "lerobot/smolvla_base",
  "camera_endpoints_by_role": {
    "camera_wrist": "https://api.cyberwave.com/api/v1/twins/{uuid}/latest-frame",
    "camera_front": "https://api.cyberwave.com/api/v1/twins/{uuid}/latest-frame"
  },
  "max_steps": 1000,
  "actions_per_cycle": 25,
  "action_sleep_seconds": 0.1,
  "inference_loop": true
}
```

---

## Output

The system produces colorized terminal output showing:

```
  Fetching weights from https://api.cyberwave.com/api/v1/mlmodels/...
  Got signed URL (expires: 2026-04-24T15:59:23+00:00)
  Downloading from signed URL...
  Extracting TAR.ZST archive...
  Resolved model directory: /root/.cache/cyberwave/weights/.../pretrained_model
  ✓ Weights downloaded to ...

══════════════════════════════════════════════════
  CYBERWAVE SETUP
══════════════════════════════════════════════════
  API Key: cw_a9ce8...4bea
  ✓ Client created
  ✓ MQTT connected
  ✓ Joints received: [-0.01, 0.01, 0.07, 0.02, -0.00, 0.12]
  ✓ Joint count matches model: 6 joints
  ✓ Camera camera_wrist ready
  ✓ Camera camera_front ready
  Started 2 background camera fetchers

══════════════════════════════════════════════════
  SMOLVLA CONTROL LOOP
══════════════════════════════════════════════════
  Max steps:        1000
  Actions/cycle:    25
  Joint source:     mqtt_subscription

──────────────────────────────────────────────────
  CYCLE 1  │  0/1000 steps (0%)
──────────────────────────────────────────────────
  Cameras: 2 frames  │  State dim: 6
  State: [-0.01, 0.01, 0.07, 0.02, -0.00, 0.12]
  Running inference...
  Predicted 50 actions -> executing 25
  [████████████████████] 25/25  ✊ [+0.11, -0.00, +0.19, +0.41, -0.15, +0.10]
  ✓ Executed 25 actions
```

Final JSON result is written to stdout for orchestration systems.

---

# Training Architecture

## Overview

The training system mirrors the inference architecture with a parallel set of components:

```
┌─────────────────────────────────────────────────────────────────┐
│                         train.py                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  • SmolVLA entry point (MODEL_SLUG = "smolvla")             ││
│  │  • Parse JSON params from Cloud Node                        ││
│  │  • Instantiate CwTrainer and run training                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓ params, model_slug               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    cw_trainer.py                            ││
│  │  • Download dataset from Cyberwave API                      ││
│  │  • Download base model weights (optional)                   ││
│  │  • Monkey-patch WandBLogger → CyberwaveLogger               ││
│  │  • Build TrainPipelineConfig via trainer registry           ││
│  │  • Execute lerobot train(), send status/metrics/ETA         ││
│  │  • Compress results to results_folder                       ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓ params                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  smolvla_trainer.py                         ││
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
│                    cw_trainer.py                            │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Cyberwave API                          │   │
│   │  ──► GET /datasets/{uuid}/zip (download dataset)    │   │
│   │  ──► GET weights_url (download base model)          │   │
│   │  ◄── PUT /mltrainings/{uuid} (metrics, ETA)         │   │
│   │  ◄── PUT /mltrainings/{uuid} (completion)           │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              SmolVLATrainer (registry lookup)       │   │
│   │  • build_pipeline_config() → TrainPipelineConfig    │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              CyberwaveLogger (monkey-patched)       │   │
│   │  • Replaces WandBLogger                             │   │
│   │  • log_dict() → PUT /mltrainings/{uuid}             │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              lerobot train(cfg)                     │   │
│   │  • Loads dataset                                    │   │
│   │  • Fine-tunes model with PEFT/LoRA                  │   │
│   │  • Saves checkpoints                                │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Results Compression                    │   │
│   │  • tar.zst checkpoint → results_folder              │   │
│   │  • Cloud Node uploads to storage                    │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
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
GET /api/v1/mlmodels/{uuid}/weights
    → Returns: { "signed_url": "...", "expires_at": "..." }

GET {signed_url}
    → Returns: tar.zst archive
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
        "estimated_end_time": "2026-01-15T18:30:00Z",
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
        "local_checkpoint_path": "./runs/artifacts/uuid.tar.zst"
    }
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
  [Cyberwave] Updated ETA: 2026-01-15T18:30:00Z (~4h 0m remaining)
  ...

--- Compression Phase ---
  Status: compressing
  Compressing ./outputs/train/abc123/checkpoints/last -> ./runs/artifacts/abc123-def456.tar.zst
  Artifact created: ./runs/artifacts/abc123-def456.tar.zst (1234.5MB)

--- Completion ---
  Completion status sent to Cyberwave

============================================================
  Training Complete
============================================================

{"status": "completed", "artifact": "./runs/artifacts/abc123-def456.tar.zst", "training_uuid": "abc123-def456"}
```
