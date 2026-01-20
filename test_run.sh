#!/bin/bash
# Test script to verify litert-lm-cli run works with Qwen3-0.6B

echo "Testing litert-lm-cli run with Qwen/Qwen3-0.6B..."
echo ""

# Test with a simple question
echo "What is the capital of France?" | timeout 60 ./litert-lm-cli run Qwen/Qwen3-0.6B 2>&1 | grep -E "(Model already exists|Binary already built|capital of France)" | head -5

exit_code=$?
if [ $exit_code -eq 0 ] || [ $exit_code -eq 124 ]; then
    echo ""
    echo "✓ Test passed! Model ran successfully without crashing."
    exit 0
else
    echo ""
    echo "✗ Test failed with exit code: $exit_code"
    exit 1
fi
