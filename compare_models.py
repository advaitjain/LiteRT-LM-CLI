#!/usr/bin/env python3
"""Compare metadata between two .litertlm models"""

import sys
import struct

def read_litertlm_metadata(filepath):
    """Read basic metadata from a .litertlm file"""
    with open(filepath, 'rb') as f:
        # Skip to section headers (at offset 16384 typically)
        f.seek(16384)

        # Try to read LLM metadata section
        # This is a simplified reader - just looking for stop token patterns
        data = f.read(100000)

        # Look for stop token marker in the metadata
        if b'stop_token' in data:
            idx = data.find(b'stop_token')
            context = data[max(0, idx-100):idx+200]
            return context

        return None

if __name__ == "__main__":
    model1 = "models/Qwen/Qwen3-0.6B/Qwen3-0.6B.litertlm"
    model2 = "models/litert-community/Qwen3-0.6B/Qwen3-0.6B.litertlm"

    print("Our converted model:")
    meta1 = read_litertlm_metadata(model1)
    if meta1:
        print(f"  Found stop_token metadata: {len(meta1)} bytes")
        # Print hex dump around stop_token
        print(f"  Context (hex): {meta1.hex()[:400]}")

    print("\nPre-converted litert-community model:")
    meta2 = read_litertlm_metadata(model2)
    if meta2:
        print(f"  Found stop_token metadata: {len(meta2)} bytes")
        print(f"  Context (hex): {meta2.hex()[:400]}")
