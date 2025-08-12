#!/usr/bin/env python3
"""
Test script to verify chat template fallback functionality
"""

import sys
from risk_inference import format_chat_messages

# Mock tokenizer class without chat template
class MockTokenizer:
    def __init__(self, has_template=False):
        self.chat_template = None if not has_template else "simple_template"
    
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if self.chat_template is None:
            raise ValueError("No chat template available")
        return "Mock template output"

def test_chat_template_fallback():
    """Test the chat template fallback functionality."""
    
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Analyze this text for security risks."},
        {"role": "assistant", "content": "I'll analyze the text for you."}
    ]
    
    print("=== Testing Chat Template Fallback ===")
    print()
    
    # Test 1: Tokenizer without chat template (should use fallback)
    print("1. Testing tokenizer WITHOUT chat template:")
    tokenizer_no_template = MockTokenizer(has_template=False)
    
    try:
        result1 = format_chat_messages(test_messages, tokenizer_no_template)
        print("✅ Fallback formatting successful!")
        print(f"Result preview: {result1[:100]}...")
        print()
    except Exception as e:
        print(f"❌ Fallback failed: {e}")
        return False
    
    # Test 2: Tokenizer with chat template (should use template)
    print("2. Testing tokenizer WITH chat template:")
    tokenizer_with_template = MockTokenizer(has_template=True)
    
    try:
        result2 = format_chat_messages(test_messages, tokenizer_with_template)
        print("✅ Template formatting successful!")
        print(f"Result: {result2}")
        print()
    except Exception as e:
        print(f"❌ Template failed: {e}")
        return False
    
    print("=== All tests passed! ===")
    return True

if __name__ == "__main__":
    success = test_chat_template_fallback()
    sys.exit(0 if success else 1)
