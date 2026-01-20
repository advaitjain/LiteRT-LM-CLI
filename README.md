# LiteRT-LM-CLI

Prototype code to make it easy to try running LLMs with LiteRT-LM

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

```bash
./litert-lm-cli run litert-community/Qwen3-0.6B
```

This will:
1. Download the model from HuggingFace (if not already cached)
2. Build the LiteRT-LM binary (if not already built)
3. Prompt you for input
4. Run inference and display the results

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
