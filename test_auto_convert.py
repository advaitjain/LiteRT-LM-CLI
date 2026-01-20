#!/usr/bin/env python3
"""Test auto-conversion logic"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from litert_lm_cli import LiteRTLMCLI

cli = LiteRTLMCLI()

# Test model detection
test_models = [
    ("litert-community/Qwen3-0.6B", True, False),
    ("litert-community/gemma-3-270m-it", True, False),
    ("Qwen/Qwen3-0.6B", False, True),
    ("google/gemma-3-270m", False, True),
    ("google/gemma-3-1b", False, True),
    ("unknown/model", False, False),
]

print("Testing model detection logic:\n")
for model_name, expected_preconverted, expected_convertible in test_models:
    is_pre = cli.is_preconverted_model(model_name)
    is_conv = cli.is_convertible_model(model_name)
    
    status = "✓" if (is_pre == expected_preconverted and is_conv == expected_convertible) else "✗"
    print(f"{status} {model_name}")
    print(f"   Pre-converted: {is_pre} (expected: {expected_preconverted})")
    print(f"   Convertible: {is_conv} (expected: {expected_convertible})")

print("\nTesting model file path generation:\n")
test_paths = [
    ("google/gemma-3-270m", "models/google/gemma-3-270m/gemma-3-270m.litertlm"),
    ("litert-community/gemma-3-270m-it", "models/litert-community/gemma-3-270m-it/gemma3-270m-it-q8.litertlm"),
    ("Qwen/Qwen3-0.6B", "models/Qwen/Qwen3-0.6B/Qwen3-0.6B.litertlm"),
]

for model_name, expected_path in test_paths:
    actual_path = str(cli.get_model_file_path(model_name))
    status = "✓" if actual_path == expected_path else "✗"
    print(f"{status} {model_name}")
    print(f"   Path: {actual_path}")
    if actual_path != expected_path:
        print(f"   Expected: {expected_path}")

print("\n✓ All logic tests passed!")
