#!/usr/bin/env python3
"""Test TFLite conversion and inference directly in Python"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "venv/lib/python3.13/site-packages"))

from ai_edge_torch.generative.examples.qwen import qwen3
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities.export_config import ExportConfig
from ai_edge_torch.generative.layers.kv_cache import KV_LAYOUT_TRANSPOSED
import transformers

def main():
    checkpoint_dir = "models/Qwen/Qwen3-0.6B/checkpoint"
    output_dir = "models/Qwen/Qwen3-0.6B"

    print("Building PyTorch model...")
    model = qwen3.build_0_6b_model(checkpoint_dir, mask_cache_size=1024)

    print("Setting up export config...")
    export_config = ExportConfig(
        kvcache_layout=KV_LAYOUT_TRANSPOSED,
        mask_as_input=True
    )

    print("Converting to TFLite (this will take a few minutes)...")
    # Convert to tflite only (not litertlm)
    converter.convert_to_tflite(
        pytorch_model=model,
        tflite_path=f"{output_dir}/test_model.tflite",
        prefill_seq_len=256,
        kv_cache_max_len=1024,
        quantize="dynamic_int8",
        export_config=export_config
    )

    print(f"\n✓ Converted to TFLite: {output_dir}/test_model.tflite")

    # Now test the TFLite model
    print("\nTesting TFLite model inference...")
    import tensorflow as tf
    import numpy as np

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=f"{output_dir}/test_model.tflite")

    # Get signatures
    signatures = interpreter.get_signature_list()
    print(f"Available signatures: {list(signatures.keys())}")

    # Test with prefill
    if 'prefill' in signatures:
        prefill_fn = interpreter.get_signature_runner('prefill')

        # Tokenize input
        prompt = "What is the capital of France?"
        tokens = tokenizer.encode(prompt, return_tensors="np")
        print(f"\nInput prompt: {prompt}")
        print(f"Input tokens: {tokens}")
        print(f"Token shape: {tokens.shape}")

        # Try to run prefill
        try:
            outputs = prefill_fn(tokens=tokens)
            print(f"\nPrefill outputs keys: {outputs.keys()}")

            # Try decode
            if 'decode' in signatures:
                decode_fn = interpreter.get_signature_runner('decode')
                print("\nDecode signature available")
        except Exception as e:
            print(f"\nError running inference: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
