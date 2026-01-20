# Gemma3 Model Support

This CLI now supports Google Gemma3 models for both running pre-converted models and converting from HuggingFace.

## Supported Models

### Pre-converted Models (litert-community)
- `litert-community/gemma-3-270m-it` - 270M instruction-tuned model

### Convertible Models (google)
- `google/gemma-3-270m` - 270M base model
- `google/gemma-3-1b` - 1B base model

## Authentication

Gemma3 models are gated and require HuggingFace authentication. Set your HuggingFace token:

```bash
export HF_TOKEN=your_huggingface_token_here
```

You can get your token from: https://huggingface.co/settings/tokens

## Running Pre-converted Models

```bash
export HF_TOKEN=your_token
./litert-lm-cli run litert-community/gemma-3-270m-it
```

Example:
```bash
$ ./litert-lm-cli run litert-community/gemma-3-270m-it
Downloading model from https://huggingface.co/litert-community/gemma-3-270m-it/...
✓ Model downloaded successfully

Enter your prompt:
> What is machine learning?

<start_of_turn>model
Machine learning is a subset of artificial intelligence...
<end_of_turn>
```

## Converting Models from HuggingFace

Convert `google/gemma-3-270m` to .litertlm format:

```bash
export HF_TOKEN=your_token
./litert-lm-cli convert google/gemma-3-270m
```

This will:
1. Download model files from HuggingFace
2. Build PyTorch model
3. Convert to TFLite with dynamic_int8 quantization
4. Package as .litertlm file

Output will be saved to: `models/google/gemma-3-270m/gemma-3-270m.litertlm`

### Conversion Parameters

```bash
./litert-lm-cli convert google/gemma-3-270m \
  --quantize dynamic_int8 \
  --kv-cache-max-len 1280 \
  --output-dir custom/path
```

Parameters:
- `--quantize`: Quantization scheme (default: dynamic_int8)
- `--kv-cache-max-len`: Maximum KV cache length (default: 1280)
- `--prefill-seq-len`: Prefill sequence length (default: 256, expands to multiple)
- `--output-dir`: Custom output directory (optional)

## Technical Details

### Tokenizer
Gemma3 uses **SentencePiece** tokenizer (not HuggingFace BPE like Qwen3).

Required files:
- `model.safetensors` - Model weights
- `config.json` - Model configuration
- `tokenizer.model` - SentencePiece tokenizer
- `tokenizer_config.json` - Tokenizer configuration

### Conversion Configuration
- **KV cache layout**: Transposed
- **Attention mask**: As input (mask_as_input=True)
- **mask_cache_size**: 0 (for mask_as_input=True)
- **Stop tokens**: [1] (EOS token)
- **Quantization**: dynamic_int8
- **Prefill lengths**: [8, 64, 128, 256, 512, 1024]

### Prompt Format
Gemma3 uses the "turn-based" chat format:

```
<start_of_turn>user
Your question here<end_of_turn>
<start_of_turn>model
Model response<end_of_turn>
```

### Sampler Parameters
Default sampler settings (from `gemma3_base_metadata.textproto`):
- Temperature: 1.0
- Top-p: 0.95
- Top-k: 40
- Max tokens: 4096

## Comparison: Qwen3 vs Gemma3

| Feature | Qwen3 | Gemma3 |
|---------|-------|--------|
| Tokenizer | HuggingFace BPE | SentencePiece |
| Stop token | 151645 (`<\|im_end\|>`) | 1 (EOS) |
| Prompt format | IM format with `<\|im_start\|>` | Turn-based with `<start_of_turn>` |
| Temperature | 0.6 | 1.0 |
| Top-k | 20 | 40 |
| Sizes | 0.6B, 1.7B, 4B | 270M, 1B |

## Example Workflow

1. **Convert the model:**
   ```bash
   export HF_TOKEN=your_token
   ./litert-lm-cli convert google/gemma-3-270m
   ```

2. **Run inference:**
   ```bash
   ./litert-lm-cli run google/gemma-3-270m
   ```

3. **Or use pre-converted:**
   ```bash
   ./litert-lm-cli run litert-community/gemma-3-270m-it
   ```

## Troubleshooting

### Error: Unauthorized (401)
You need to set HF_TOKEN for gated models:
```bash
export HF_TOKEN=your_token
```

### Model not found
Make sure you've accepted the license agreement on HuggingFace for Gemma models.

### Conversion fails
Ensure you have the latest ai-edge-torch-nightly installed:
```bash
pip install --upgrade ai-edge-torch-nightly
```
