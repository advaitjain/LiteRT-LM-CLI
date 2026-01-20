#!/usr/bin/env python3
"""Inspect .litertlm model metadata"""

import sys
from pathlib import Path

# Add ai_edge_litert to path
sys.path.insert(0, str(Path(__file__).parent / "venv/lib/python3.13/site-packages"))

from ai_edge_litert.internal import litertlm_core

def inspect_model(filepath):
    """Inspect a .litertlm model file"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*60}")

    try:
        # Open the model file
        model = litertlm_core.open_litertlm_file(filepath)

        # Get LLM metadata
        llm_metadata = model.get_llm_metadata()

        print("\nLLM Metadata:")
        print(f"  Model type: {llm_metadata.llm_model_type}")
        print(f"  Max tokens: {llm_metadata.max_num_tokens}")

        # Check start token
        if llm_metadata.HasField('start_token'):
            start = llm_metadata.start_token
            if start.HasField('token_str'):
                print(f"  Start token (str): {start.token_str}")
            if start.token_ids and start.token_ids.ids:
                print(f"  Start token (ids): {list(start.token_ids.ids)}")

        # Check stop tokens
        print(f"  Number of stop tokens: {len(llm_metadata.stop_tokens)}")
        for i, stop in enumerate(llm_metadata.stop_tokens):
            if stop.HasField('token_str'):
                print(f"    Stop token {i} (str): {stop.token_str}")
            if stop.token_ids and stop.token_ids.ids:
                print(f"    Stop token {i} (ids): {list(stop.token_ids.ids)}")

        # Check sampler params
        if llm_metadata.HasField('sampler_params'):
            print(f"\n  Sampler params:")
            print(f"    Type: {llm_metadata.sampler_params.type}")
            print(f"    Temperature: {llm_metadata.sampler_params.temperature}")
            print(f"    Top-p: {llm_metadata.sampler_params.p}")
            print(f"    Top-k: {llm_metadata.sampler_params.k}")

        # Check prompt template
        if llm_metadata.jinja_prompt_template:
            print(f"\n  Jinja template length: {len(llm_metadata.jinja_prompt_template)} chars")
            print(f"  Template preview: {llm_metadata.jinja_prompt_template[:150]}...")

    except Exception as e:
        print(f"Error inspecting model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_model(sys.argv[1])
    else:
        # Inspect both models
        inspect_model("models/Qwen/Qwen3-0.6B/Qwen3-0.6B.litertlm")
        inspect_model("models/litert-community/Qwen3-0.6B/Qwen3-0.6B.litertlm")
