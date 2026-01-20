# Qwen3-0.6B Conversion Status

## ✓ SUCCESS - Conversion Working Correctly

The `convert` subcommand is now fully functional and produces high-quality model outputs that match the litert-community reference implementation.

## What Works

**Our converted model** (`Qwen/Qwen3-0.6B`):
```bash
./litert-lm-cli convert Qwen/Qwen3-0.6B
printf "What is the capital of France?\n" | ./litert-lm-cli run Qwen/Qwen3-0.6B
```

Output:
```
<think>
Okay, the user is asking about the capital of France. I remember that France is a country
located in Northwestern Europe, and the capital is Paris. But I need to make sure I'm correct
here. Let me think. Paris is indeed the capital and the most populous city in France. I should
confirm if there's any other city that might be considered the capital. No, I don't recall any
other major cities having that status. So the answer should be Paris.
</think>

The capital of France is Paris.
```

**Pre-converted reference** (`litert-community/Qwen3-0.6B`):
```bash
printf "What is the capital of France?\n" | ./litert-lm-cli run litert-community/Qwen3-0.6B
```

Output:
```
<think>
Okay, the user is asking for the capital of France. I need to make sure I recall the correct
answer. France's capital is Paris. Let me think... I remember that Paris is the largest city
in France and is the political and cultural center of the country. I should confirm that there
isn't any other city that is considered the capital. For example, maybe some other city has a
government structure but is not the official capital. But I think Paris is definitely the
answer here. I should also mention that it's the most populated city in the world, which adds
context. Let me check if there's any possibility of confusion, but I don't think so. So the
answer should be Paris.
</think>

The capital of France is **Paris**.
```

Both models produce high-quality output with proper reasoning and clean answers!

## Critical Conversion Parameters

The following parameters were essential for successful conversion:

### 1. Model Building Parameters
```python
model = qwen3.build_0_6b_model(
    checkpoint_path=checkpoint_dir,
    custom_loader=None,
    mask_cache_size=0  # Critical: 0 when mask_as_input=True
)
```

### 2. Export Configuration
```python
export_config = ExportConfig(
    kvcache_layout=KV_LAYOUT_TRANSPOSED,
    mask_as_input=True
)
```

### 3. Conversion Parameters
```python
converter.convert_to_litert(
    pytorch_model=model,
    prefill_seq_len=[8, 64, 128, 256, 512, 1024],  # Multiple prefill signatures
    kv_cache_max_len=1280,  # Not 1024
    quantize="dynamic_int8",
    export_config=export_config,

    # Tokenizer
    hf_tokenizer_model_path=str(tokenizer_path),  # tokenizer.json file

    # Stop tokens
    stop_token_ids=[151645],  # <|im_end|> token

    # Prompt templates (IM format)
    user_prompt_prefix="<|im_start|>user\n",
    user_prompt_suffix="<|im_end|>\n",
    model_prompt_prefix="<|im_start|>assistant\n",
    model_prompt_suffix="<|im_end|>\n",

    # Chat template
    jinja_prompt_template=<simplified_template>,

    # Sampler configuration
    base_llm_metadata_path=str(base_metadata_path)  # temperature=0.6, top_p=0.95, top_k=20
)
```

## Key Fixes That Resolved Output Quality Issues

| Issue | Solution | Impact |
|-------|----------|--------|
| Repetitive/looping text | Set `temperature=0.6` (not 1.0) via base_metadata.textproto | High - Controls output diversity |
| Missing reasoning tags | Add prompt templates with `<\|im_start\|>` and `<\|im_end\|>` | High - Proper message formatting |
| Incorrect generation | Use multiple `prefill_seq_len` values [8, 64, 128, 256, 512, 1024] | Medium - Better input handling |
| Wrong cache size | Set `mask_cache_size=0` when `mask_as_input=True` | High - Correct attention computation |
| Short max length | Use `kv_cache_max_len=1280` (not 1024) | Low - Allows longer generations |

## File Size Comparison

| Model | Size | Notes |
|-------|------|-------|
| litert-community/Qwen3-0.6B | 614 MB | Reference implementation |
| Our Qwen/Qwen3-0.6B | 619 MB | Successfully converted (5MB difference is acceptable) |

## Implementation Files

- `converter/__init__.py` - Package initialization
- `converter/base.py` - Abstract converter base class
- `converter/download.py` - HuggingFace download utilities
- `converter/qwen3_converter.py` - Qwen3 converter implementation
- `converter/qwen3_base_metadata.textproto` - Sampler parameters (temperature, top_p, top_k)
- `litert-lm-cli` - Main CLI with convert subcommand

## Usage

### Convert a Model
```bash
./litert-lm-cli convert Qwen/Qwen3-0.6B
```

Optional parameters:
```bash
./litert-lm-cli convert Qwen/Qwen3-0.6B \
  --quantize dynamic_int8 \
  --kv-cache-max-len 1280 \
  --output-dir custom/path
```

### Run Converted Model
```bash
./litert-lm-cli run Qwen/Qwen3-0.6B
```

The run command automatically uses the locally converted model if it exists, otherwise downloads from HuggingFace.

## Supported Models

Currently supported Qwen3 models:
- Qwen/Qwen3-0.6B ✓
- Qwen/Qwen3-1.7B (should work with same parameters)
- Qwen/Qwen3-4B (should work with same parameters)

## Future Extensions

The modular converter architecture allows easy addition of:
- Gemma3 models (SentencePiece tokenizer)
- Llama models
- Other HuggingFace transformers models
- Custom quantization schemes
- Different backend optimizations
