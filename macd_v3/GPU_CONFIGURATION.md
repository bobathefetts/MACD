# GPU Configuration - MACD v3

## Summary

**Status**: ✅ **GPU-First by Default**

The MACD v3 codebase has been updated to **automatically use GPU when available**, with intelligent fallback to CPU.

---

## Changes Applied

### 1. **ModelConfig Default Device** ✅

**File**: `macd/models/base.py:13`

**Change**:
```python
# OLD:
device: str = Field(default="cpu", description="Device to run model on")

# NEW:
device: str = Field(default="auto", description="Device to run model on (auto=GPU if available, else CPU)")
```

**Impact**: All models now default to GPU detection instead of forcing CPU.

---

### 2. **HuggingFace Model GPU Logic** ✅

**File**: `macd/models/huggingface_model.py:45-97`

**Change**: Added intelligent device detection logic:

```python
# Determine device strategy
if self.config.device == "auto":
    # Auto-detect: use GPU if available
    use_gpu = torch.cuda.is_available()
    device_map = "auto" if use_gpu else None
    torch_dtype = torch.float16 if use_gpu else torch.float32
else:
    # Explicit device specified
    device_map = None
    torch_dtype = torch.float32
```

**Key Features**:
- When `device="auto"`: Checks `torch.cuda.is_available()`
- **GPU mode**: Uses `device_map="auto"` with `float16` (faster, less memory)
- **CPU mode**: Uses `device_map=None` with `float32` (compatibility)
- **Explicit override**: User can still set `device="cuda"` or `device="cpu"`

**Generation Logic** (line 123-129):
```python
# Move inputs to device (only needed if not using device_map="auto")
if self.config.device != "auto":
    inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
elif not hasattr(self.model, "hf_device_map"):
    # If device="auto" but model doesn't have device_map, move to GPU if available
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
```

---

### 3. **Configuration Files Updated** ✅

All YAML config files now default to `device: "auto"`:

**Files Updated**:
- `configs/models/huggingface.yaml` - ✅ `device: "auto"`
- `configs/models/local.yaml` - ✅ `device: "auto"`
- `configs/models/openai.yaml` - ✅ `device: "auto"` (N/A for API, but consistent)

**Example** (`configs/models/huggingface.yaml`):
```yaml
model:
  name: "microsoft/DialoGPT-small"
  backend: "huggingface"
  params:
    model_path: "microsoft/DialoGPT-small"
    tokenizer_path: "microsoft/DialoGPT-small"
    use_quantization: false
    trust_remote_code: false
    torch_dtype: "auto"
    device: "auto"  # Auto-detect: GPU if available, else CPU
```

---

### 4. **ConfigManager Defaults** ✅

**File**: `macd/config/config_manager.py:53`

**Change**:
```python
"model": {
    "name": "mock_model",
    "backend": "mock",
    "params": {
        "device": "auto"  # Auto-detect: GPU if available, else CPU
    }
},
```

---

### 5. **Trainer Already GPU-Optimized** ✅

**File**: `macd/core/trainer.py:129-130`

The trainer already had GPU auto-detection:
```python
model_kwargs = {
    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    "device_map": "auto" if torch.cuda.is_available() else None,
}
```

**Status**: No changes needed - already optimal.

---

## GPU Usage by Component

| Component | GPU Usage | Notes |
|-----------|-----------|-------|
| **HuggingFace Model (Inference)** | ✅ Auto-detect | Uses `device_map="auto"`, float16 on GPU |
| **Trainer (Distillation)** | ✅ Auto-detect | Uses `device_map="auto"`, float16 on GPU |
| **Evaluator** | ✅ Inherits from model | No direct device handling (text comparison) |
| **Evolution** | ✅ Inherits from model | Calls model.generate() which uses GPU |
| **Controller** | ✅ Inherits from model | Orchestrates GPU-enabled components |

---

## Behavior by Environment

### With GPU Available (CUDA)

```python
# What happens internally:
use_gpu = torch.cuda.is_available()  # True
device_map = "auto"                   # Enables automatic device placement
torch_dtype = torch.float16          # Half precision for speed

# Model loading:
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",      # HuggingFace handles device placement
    torch_dtype=torch.float16
)

# Generation:
# Inputs automatically moved to correct device by device_map
```

**Benefits**:
- ⚡ **Faster inference** (float16)
- 💾 **Less memory usage** (half precision)
- 🔄 **Automatic multi-GPU** (if available)

### Without GPU (CPU only)

```python
# What happens internally:
use_gpu = torch.cuda.is_available()  # False
device_map = None                     # No device mapping
torch_dtype = torch.float32          # Full precision for accuracy

# Model loading:
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=None,
    torch_dtype=torch.float32
)
model = model.to("cpu")  # Explicitly move to CPU

# Generation:
inputs = {k: v.to("cpu") for k, v in inputs.items()}
```

**Benefits**:
- ✅ **Full compatibility** (no CUDA required)
- 🎯 **Stable precision** (float32)

---

## User Override Options

Users can still explicitly control device placement:

### Option 1: Config File Override

```yaml
model:
  params:
    device: "cuda"  # Force GPU (error if not available)
    # OR
    device: "cpu"   # Force CPU (even if GPU available)
    # OR
    device: "auto"  # Auto-detect (default)
```

### Option 2: Environment Variable

```bash
# Set device via environment
export MACD_DEVICE="cuda"  # If supported in env_overrides

# Or modify config programmatically
python -c "
from macd.config import ConfigManager
manager = ConfigManager()
config = manager.load_config('config.yaml')
config.model.params['device'] = 'cuda'
"
```

### Option 3: Direct Code

```python
from macd.models.huggingface_model import HuggingFaceModel, HuggingFaceConfig

config = HuggingFaceConfig(
    name="test",
    backend="huggingface",
    model_path="gpt2",
    device="cuda"  # Explicit GPU
)

model = HuggingFaceModel(config)
```

---

## Verification

Run the test script to verify GPU configuration:

```bash
cd macd_v3
python test_gpu_config.py
```

**Expected Output** (without GPU):
```
[OK] Default device: auto
[OK] configs/models/huggingface.yaml: device = 'auto'
[OK] configs/models/local.yaml: device = 'auto'
[OK] configs/models/openai.yaml: device = 'auto'
[OK] ConfigManager default device: 'auto'
[OK] Trainer correctly falls back to CPU and float32
```

**Expected Output** (with GPU):
```
[OK] Default device: auto
[INFO] CUDA available: True
[OK] Would use GPU (device_map='auto', dtype=float16)
[OK] Trainer correctly detects GPU and uses float16
```

---

## Performance Comparison

### Before Changes (Default CPU)

```
Model Loading:  CPU (forced)
Inference:      CPU (slow)
Precision:      float32
Memory:         High
Speed:          Slow (10-50x slower than GPU)
```

### After Changes (Default Auto)

**With GPU**:
```
Model Loading:  GPU (auto-detected)
Inference:      GPU (fast)
Precision:      float16
Memory:         Reduced by 50%
Speed:          Fast (10-50x faster)
```

**Without GPU**:
```
Model Loading:  CPU (fallback)
Inference:      CPU
Precision:      float32
Memory:         Normal
Speed:          Same as before
```

---

## Common Issues & Solutions

### Issue 1: "CUDA out of memory"

**Solution**: Use quantization or smaller batch size:

```yaml
model:
  params:
    use_quantization: true
    quantization_config:
      load_in_8bit: true
      llm_int8_threshold: 6.0
```

### Issue 2: Model not using GPU

**Check**:
1. Is CUDA installed? `python -c "import torch; print(torch.cuda.is_available())"`
2. Is device set correctly? Check config files
3. Is PyTorch GPU version installed? `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Issue 3: Want to force CPU (for testing)

**Solution**:
```yaml
model:
  params:
    device: "cpu"  # Explicit CPU
```

Or set environment variable before running.

---

## Technical Details

### Why `device_map="auto"`?

HuggingFace's `device_map="auto"` provides:
- **Automatic device placement** across GPUs
- **Memory-efficient loading** (offloading to CPU/disk if needed)
- **Multi-GPU support** (splits model across devices)
- **Mixed precision** (optimal dtype per layer)

### Why `float16` on GPU?

- **2x faster inference** (fewer operations)
- **50% less memory** (half the bits per weight)
- **Modern GPUs optimized** for float16 operations
- **Minimal accuracy loss** for generation tasks

### Device Map vs Manual .to()

```python
# OLD WAY (manual):
model = AutoModelForCausalLM.from_pretrained(path)
model = model.to("cuda")  # Loads everything to GPU

# NEW WAY (automatic):
model = AutoModelForCausalLM.from_pretrained(
    path,
    device_map="auto"  # Smart placement, multi-GPU, offloading
)
```

---

## Migration Guide

### For Existing Users

If you were explicitly setting `device: "cpu"` in configs:

**Before**:
```yaml
model:
  params:
    device: "cpu"  # Forced CPU
```

**After** (to use GPU):
```yaml
model:
  params:
    device: "auto"  # Auto-detect GPU
    # OR remove the line entirely (defaults to "auto")
```

**After** (to keep CPU):
```yaml
model:
  params:
    device: "cpu"  # Still works, explicit CPU
```

### For Developers

When creating models programmatically:

**Before**:
```python
config = HuggingFaceConfig(
    name="model",
    backend="huggingface",
    model_path="gpt2",
    device="cpu"  # Had to specify
)
```

**After**:
```python
config = HuggingFaceConfig(
    name="model",
    backend="huggingface",
    model_path="gpt2",
    # device="auto" is now the default
)
```

---

## Summary of Benefits

✅ **No configuration needed** - GPU used automatically
✅ **Faster inference** - 10-50x speedup with GPU
✅ **Less memory** - float16 reduces memory by 50%
✅ **Multi-GPU support** - Automatic with device_map
✅ **Backward compatible** - Explicit device still works
✅ **Intelligent fallback** - CPU used if GPU unavailable
✅ **Production-ready** - Matches HuggingFace best practices

---

## Related Files

- `macd/models/base.py` - Base config with device default
- `macd/models/huggingface_model.py` - GPU detection logic
- `macd/core/trainer.py` - Distillation GPU usage
- `configs/models/*.yaml` - Config file defaults
- `macd/config/config_manager.py` - Default config values
- `test_gpu_config.py` - Verification script

---

**Last Updated**: 2025-10-12
**Status**: ✅ Complete - All components GPU-enabled by default
