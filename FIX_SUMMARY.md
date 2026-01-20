# Fix Summary: Stop Token Configuration for Qwen3 Models

## Problem
The `litert-lm-cli run Qwen/Qwen3-0.6B` command was crashing with the error:
```
Stop tokens are required. Either set the stop token ids or provide a valid stop token in the model file/engine settings.
```

## Root Cause
The converted `.litertlm` model file was missing stop token configuration in its metadata. The LiteRT-LM runtime requires stop tokens to know when to terminate text generation.

## Solution
Updated `converter/qwen3_converter.py` to include stop token IDs when converting the model:

### Changes Made
1. Identified the correct stop token for Qwen3 models:
   - Token: `<|im_end|>`
   - Token ID: `151645`
   - Source: `tokenizer_config.json` from the HuggingFace model

2. Modified the `convert()` method in `Qwen3Converter` to pass stop tokens:
   ```python
   # Qwen3 uses <|im_end|> as the stop token (token ID: 151645)
   stop_token_ids = [151645]  # <|im_end|>

   converter.convert_to_litert(
       pytorch_model=model,
       output_path=str(self.output_dir),
       output_name_prefix=self.short_model_name,
       prefill_seq_len=prefill_seq_len,
       kv_cache_max_len=kv_cache_max_len,
       quantize=quantize,
       export_config=export_config,
       output_format="litertlm",
       hf_tokenizer_model_path=str(tokenizer_path),
       stop_token_ids=stop_token_ids  # <-- Added this parameter
   )
   ```

## Verification
After reconverting the model with stop tokens:
- ✓ Model loads successfully
- ✓ Stop tokens are properly configured in model metadata
- ✓ Inference runs without crashing
- ✓ Model generates text output

## Test Command
```bash
echo "What is the capital of France?" | ./litert-lm-cli run Qwen/Qwen3-0.6B
```

This now works without crashing and properly terminates generation at the stop token.

## Files Modified
- `converter/qwen3_converter.py` (line 151, 171)
