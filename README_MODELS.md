# Supported Models

This CLI supports multiple model families for both running pre-converted models and converting from HuggingFace.

## Quick Reference

| Model Family | Pre-converted | Convert from HF | Tokenizer | Auth Required |
|--------------|---------------|-----------------|-----------|---------------|
| **Qwen3** | litert-community/Qwen3-0.6B | Qwen/Qwen3-{0.6B,1.7B,4B} | HuggingFace BPE | No |
| **Gemma3** | litert-community/gemma-3-270m-it | google/gemma-3-{270m,270m-it,1b} | SentencePiece | Yes (HF_TOKEN) |

## Qwen3 Models

### Pre-converted
```bash
./litert-lm-cli run litert-community/Qwen3-0.6B
```

### Convert from HuggingFace
```bash
./litert-lm-cli convert Qwen/Qwen3-0.6B
./litert-lm-cli convert Qwen/Qwen3-1.7B
./litert-lm-cli convert Qwen/Qwen3-4B
```

### Technical Details
- **Tokenizer**: HuggingFace BPE (tokenizer.json)
- **Stop token**: 151645 (`<|im_end|>`)
- **Prompt format**: `<|im_start|>role\n...content...<|im_end|>\n`
- **Temperature**: 0.6
- **Top-k**: 20
- **Documentation**: See [CONVERSION_STATUS.md](CONVERSION_STATUS.md)

## Gemma3 Models

### Pre-converted
```bash
export HF_TOKEN=your_token
./litert-lm-cli run litert-community/gemma-3-270m-it
```

### Convert from HuggingFace
```bash
export HF_TOKEN=your_token
source venv/bin/activate
python3 litert-lm-cli convert google/gemma-3-270m-it  # ✅ Instruction-tuned (recommended for chat)
python3 litert-lm-cli convert google/gemma-3-270m     # Base model (for completion tasks)
python3 litert-lm-cli convert google/gemma-3-1b
```

### Technical Details
- **Tokenizer**: SentencePiece (tokenizer.model)
- **Start token**: 2 (BOS)
- **Stop tokens**: 1 (EOS), 106 (`<end_of_turn>`)
- **Prompt format**: `<start_of_turn>role\n...content...<end_of_turn>\n`
- **Temperature**: 1.0
- **Top-k**: 1 (greedy decoding for best quality)
- **Top-p**: 0.95
- **Max tokens**: 4096
- **Documentation**: See [CONVERSION_STATUS.md](CONVERSION_STATUS.md)

### Authentication
Gemma3 models are gated. You need to:
1. Get token: https://huggingface.co/settings/tokens
2. Accept license: https://huggingface.co/google/gemma-3-270m
3. Set environment: `export HF_TOKEN=your_token`

## Common Usage Patterns

### Run a pre-converted model
```bash
./litert-lm-cli run MODEL_NAME
```

### Convert and run
```bash
./litert-lm-cli convert MODEL_NAME
./litert-lm-cli run MODEL_NAME
```

### Advanced conversion
```bash
./litert-lm-cli convert MODEL_NAME \
  --quantize dynamic_int8 \
  --kv-cache-max-len 1280 \
  --output-dir /custom/path
```

## Model Sizes

| Model | Parameters | Output File Size (dynamic_int8) |
|-------|-----------|----------------------------------|
| Qwen3-0.6B | 600M | 619 MB |
| Qwen3-1.7B | 1.7B | ~1.7 GB (estimated) |
| Qwen3-4B | 4B | ~4 GB (estimated) |
| Gemma3-270m-it | 270M | 299 MB |
| Gemma3-270m | 270M | 299 MB |
| Gemma3-1b | 1B | ~1 GB (estimated) |

## File Locations

After conversion, models are saved to:
```
models/
├── Qwen/
│   └── Qwen3-0.6B/
│       ├── checkpoint/           # Downloaded HF files
│       └── Qwen3-0.6B.litertlm
├── google/
│   ├── gemma-3-270m/
│   │   ├── checkpoint/           # Downloaded HF files
│   │   └── gemma-3-270m.litertlm
│   └── gemma-3-270m-it/
│       ├── checkpoint/           # Downloaded HF files
│       └── gemma-3-270m-it.litertlm
└── litert-community/
    ├── Qwen3-0.6B/
    │   └── Qwen3-0.6B.litertlm
    └── gemma-3-270m-it/
        └── gemma3-270m-it-q8.litertlm
```

## Conversion Parameters

All models support these conversion options:

- `--quantize`: Quantization type (default: `dynamic_int8`)
  - Options: `dynamic_int8`, `int8`, `fp16`

- `--kv-cache-max-len`: Maximum KV cache length (default: `1280`)
  - Affects max generation length

- `--prefill-seq-len`: Prefill sequence length (default: `256`)
  - Automatically expands to multiple values: [8, 64, 128, 256, 512, 1024]

- `--output-dir`: Custom output directory (optional)
  - Default: `models/{org}/{model_name}/`

## Error Messages

### "Unauthorized (401)"
Set HF_TOKEN for gated models:
```bash
export HF_TOKEN=your_huggingface_token
```

### "Model already exists"
The model is already downloaded/converted. Delete it to re-convert:
```bash
rm models/path/to/model.litertlm
```

### "Build failed"
Ensure LiteRT-LM binary is built:
```bash
./setup
```

## Adding New Models

To add support for a new model family:

1. Create `converter/model_converter.py` extending `ModelConverter`
2. Create `converter/model_base_metadata.textproto` with sampler params
3. Add detection logic in `litert-lm-cli` `convert_model()` method
4. Add special filename mapping if needed in `download_model()`

See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for details.

## Further Documentation

- **Model Conversion**: [CONVERSION_STATUS.md](CONVERSION_STATUS.md) - Covers both Qwen3 and Gemma3
- **Implementation Details**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Auto-conversion**: [AUTO_CONVERSION.md](AUTO_CONVERSION.md)
