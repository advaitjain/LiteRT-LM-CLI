# LiteRT-LM-CLI

Prototype code to make it easy to try running LLMs with LiteRT-LM

## Supported Models

| Model Family | Pre-converted | Convert from HF | Size | Status |
|--------------|---------------|-----------------|------|--------|
| **Qwen3** | litert-community/Qwen3-0.6B | Qwen/Qwen3-{0.6B,1.7B,4B} | 619 MB | ✅ Fully tested |
| **Gemma3** | litert-community/gemma-3-270m-it | google/gemma-3-{270m-it,270m,1b} | 299 MB | ✅ Fully tested |

**Note**: Gemma3 models require HuggingFace authentication (`HF_TOKEN`).

For detailed model information, see [README_MODELS.md](README_MODELS.md).

## Setup

Run the setup script to prepare the environment:

```bash
./setup
```

This will:
1. Create a Python virtual environment
2. Install `ai-edge-torch-nightly` and `ai-edge-litert-nightly`
3. Clone the LiteRT-LM repository and checkout the latest release
4. Download and configure Bazelisk

## Usage

Start with:
```bash
source venv/bin/activate
```


### Help

```bash
./litert-lm-cli -h
```

### Run a Model

Pre-converted models:
```bash
source venv/bin/activate
./litert-lm-cli run litert-community/Qwen3-0.6B
./litert-lm-cli run litert-community/gemma-3-270m-it
```

Auto-conversion (downloads and converts from HuggingFace on first run):
```bash
source venv/bin/activate
./litert-lm-cli run Qwen/Qwen3-0.6B
./litert-lm-cli run google/gemma-3-270m-it  # Requires HF_TOKEN
```

This will:
1. Download the model from HuggingFace (if not already cached)
2. Convert to LiteRT-LM format (if needed)
3. Build the LiteRT-LM binary (if not already built)
4. Prompt you for input
5. Run inference and display the results

### Convert a Model

Explicitly convert a model before running:
```bash
source venv/bin/activate
./litert-lm-cli convert google/gemma-3-270m-it
./litert-lm-cli convert Qwen/Qwen3-0.6B
```

## Directory Structure

```
LiteRT-LM-CLI/
├── setup              # Setup script
├── litert-lm-cli     # Main CLI tool
├── venv/             # Python virtual environment
├── LiteRT-LM/        # Cloned LiteRT-LM repository
├── bazelisk          # Bazelisk binary
└── models/           # Downloaded models
```

## Requirements

- Python 3.8+
- Git
- curl
- Linux or macOS

All dependencies are installed within the LiteRT-LM-CLI directory and do not interfere with the system.

## Documentation

- **[README_MODELS.md](README_MODELS.md)** - Comprehensive model reference guide
  - All supported models and variants
  - Technical details (tokenizers, stop tokens, prompt formats)
  - File locations and sizes
  - Authentication requirements

- **[CONVERSION_STATUS.md](CONVERSION_STATUS.md)** - Model conversion details
  - Conversion status for all models
  - Critical configuration parameters
  - Troubleshooting and examples
  - Quality comparisons with reference implementations
