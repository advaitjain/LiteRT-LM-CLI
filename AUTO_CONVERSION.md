# Auto-Conversion Feature

The `litert-lm-cli run` command now automatically converts models from HuggingFace if they haven't been converted yet.

## How It Works

When you run `./litert-lm-cli run MODEL_NAME`, the CLI:

1. **Checks if model exists locally**
   - If `.litertlm` file exists → runs directly
   - If not → proceeds to step 2

2. **Determines model type**
   - **Pre-converted models** (litert-community/*): Downloads .litertlm file from HuggingFace
   - **Convertible models** (google/*, Qwen/*): Auto-converts from source
   - **Unsupported models**: Shows error with list of supported models

3. **Auto-converts if needed**
   - Downloads checkpoint files from HuggingFace
   - Builds PyTorch model
   - Converts to TFLite with quantization
   - Packages as .litertlm file
   - Then runs inference

## Examples

### Example 1: Pre-converted Model
```bash
$ ./litert-lm-cli run litert-community/gemma-3-270m-it
Model not found locally. Downloading from HuggingFace...
Downloading model from https://huggingface.co/litert-community/gemma-3-270m-it/...
✓ Model downloaded successfully

Enter your prompt:
> Hello!
```

### Example 2: Auto-conversion
```bash
$ ./litert-lm-cli run google/gemma-3-270m
Model not found locally. Converting from HuggingFace...
This will download ~1-2GB and take several minutes.

Converting google/gemma-3-270m...
Model size: 270m
Quantization: dynamic_int8
...
[Conversion process runs]
...
✓ Model successfully converted!

Enter your prompt:
> What is machine learning?
```

### Example 3: Already Converted
```bash
$ ./litert-lm-cli run google/gemma-3-270m
✓ Model already exists at models/google/gemma-3-270m/gemma-3-270m.litertlm
✓ Binary already built

Enter your prompt:
>
```

## Authentication

### HF_TOKEN Setup

For gated models (like Gemma3), you need a HuggingFace token. The CLI checks for the token in this order:

1. **Environment variable**: `export HF_TOKEN=your_token`
2. **HF_TOKEN file** in current directory
3. **HF_TOKEN file** in parent directory

#### Option 1: Environment Variable
```bash
export HF_TOKEN=hf_your_token_here
./litert-lm-cli run google/gemma-3-270m
```

#### Option 2: Token File
```bash
# Create token file
echo "hf_your_token_here" > HF_TOKEN

# Run without export
./litert-lm-cli run google/gemma-3-270m
```

### Accepting License Agreements

Many models (especially Gemma) require accepting a license agreement:

1. Visit the model page: https://huggingface.co/google/gemma-3-270m
2. Click "Agree and access repository"
3. Wait a few minutes for access to be granted
4. Run the conversion

If you haven't accepted the license, you'll see:
```
✗ Cannot access gated model google/gemma-3-270m.

This model requires accepting a license agreement.
Please:
  1. Visit: https://huggingface.co/google/gemma-3-270m
  2. Click 'Agree and access repository'
  3. Wait a few minutes for access to be granted
  4. Try the conversion again
```

## Supported Models

### Pre-converted (Download Only)
- `litert-community/Qwen3-0.6B`
- `litert-community/gemma-3-270m-it`

### Convertible (Auto-convert)
- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B`
- `google/gemma-3-270m` (requires auth)
- `google/gemma-3-1b` (requires auth)

## Conversion Parameters

Auto-conversion uses these default parameters:
- **Quantization**: dynamic_int8
- **KV cache max length**: 1280
- **Prefill sequence lengths**: [8, 64, 128, 256, 512, 1024]

For custom parameters, use the `convert` command explicitly:
```bash
./litert-lm-cli convert google/gemma-3-270m \
  --quantize dynamic_int8 \
  --kv-cache-max-len 2048 \
  --output-dir /custom/path
```

## File Locations

Converted models are saved to:
```
models/
├── google/
│   └── gemma-3-270m/
│       ├── checkpoint/              # Downloaded HF files
│       │   ├── model.safetensors    (~1GB)
│       │   ├── config.json
│       │   ├── tokenizer.model
│       │   └── tokenizer_config.json
│       └── gemma-3-270m.litertlm    (~270MB)
│
├── Qwen/
│   └── Qwen3-0.6B/
│       ├── checkpoint/              # Downloaded HF files
│       └── Qwen3-0.6B.litertlm     (~619MB)
│
└── litert-community/
    └── gemma-3-270m-it/
        └── gemma3-270m-it-q8.litertlm
```

## Conversion Time and Size

| Model | Download Size | Conversion Time | Output Size |
|-------|---------------|-----------------|-------------|
| google/gemma-3-270m | ~1.0 GB | ~5-10 min | ~270 MB |
| google/gemma-3-1b | ~2.0 GB | ~10-15 min | ~1 GB |
| Qwen/Qwen3-0.6B | ~1.2 GB | ~10-15 min | ~619 MB |
| Qwen/Qwen3-1.7B | ~3.4 GB | ~15-25 min | ~1.7 GB |

*Times are approximate and depend on hardware (CPU, RAM) and network speed.*

## Workflow Comparison

### Old Workflow (Manual)
```bash
# Step 1: Convert
./litert-lm-cli convert google/gemma-3-270m

# Step 2: Run
./litert-lm-cli run google/gemma-3-270m
```

### New Workflow (Auto)
```bash
# One command - auto-converts if needed
./litert-lm-cli run google/gemma-3-270m
```

## Troubleshooting

### "No module named 'huggingface_hub'"
The script needs to use the virtual environment. Make sure the shebang points to venv:
```bash
#!/home/path/to/LiteRT-LM-CLI/venv/bin/python3
```

### "Cannot access gated repo"
1. Check HF_TOKEN is set (or file exists)
2. Accept license agreement on HuggingFace
3. Wait a few minutes for access to be granted
4. Try again

### "Conversion completed but model file not found"
This indicates a bug in the conversion process. Check:
1. Disk space (need ~3GB free)
2. Conversion logs for errors
3. Try manual conversion: `./litert-lm-cli convert MODEL_NAME`

### Model Already Exists but Need to Reconvert
Delete the old .litertlm file:
```bash
rm models/google/gemma-3-270m/gemma-3-270m.litertlm
./litert-lm-cli run google/gemma-3-270m
```

## Benefits

### Simplified Workflow
- Single command for both new and existing models
- No need to remember whether a model has been converted
- Automatic detection and handling

### Better UX
- Clear progress messages during conversion
- Helpful error messages with actionable steps
- Auto-detects token from file or environment

### Developer Friendly
- Token file support (no need to export every time)
- Reuses existing .litertlm files (no unnecessary reconversions)
- Works with all supported models

## Implementation Details

The auto-conversion feature is implemented in the `run_model()` method:

1. **Model Detection**
   ```python
   def is_preconverted_model(self, model_name):
       return model_name.startswith("litert-community/")

   def is_convertible_model(self, model_name):
       # Checks for Qwen3 and Gemma3 patterns
       # Excludes pre-converted models
   ```

2. **Token Resolution**
   ```python
   def get_hf_token():
       # 1. Check HF_TOKEN environment variable
       # 2. Check HF_TOKEN file in current dir
       # 3. Check HF_TOKEN file in parent dir
   ```

3. **Auto-conversion Flow**
   ```python
   if not model_file.exists():
       if is_preconverted_model(model_name):
           download_model(model_name)
       elif is_convertible_model(model_name):
           convert_model(model_name)  # Auto-convert
       else:
           show_error_with_supported_models()
   ```

## See Also

- [GEMMA3_SUPPORT.md](GEMMA3_SUPPORT.md) - Gemma3-specific documentation
- [CONVERSION_STATUS.md](CONVERSION_STATUS.md) - Qwen3 conversion details
- [README_MODELS.md](README_MODELS.md) - Complete model reference
