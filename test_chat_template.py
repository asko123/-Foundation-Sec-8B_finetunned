#!/usr/bin/env python3
"""
Test script to verify chat template fallback functionality
"""

from risk_inference import format_messages_for_model

class MockTokenizer:
    """Mock tokenizer without chat template"""
    def __init__(self):
        self.chat_template = None

def test_fallback_formatting():
    """Test that fallback formatting works when chat template is not available"""
    
    tokenizer = MockTokenizer()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Analyze this text for security risks."},
        {"role": "assistant", "content": "I'll analyze the text for you."}
    ]
    
    result = format_messages_for_model(messages, tokenizer)
    
    print("✅ Chat template fallback test:")
    print("=" * 50)
    print(result)
    print("=" * 50)
    
    # Verify the formatting
    assert "System:" in result
    assert "User:" in result  
    assert "Assistant:" in result
    print("✅ All formatting checks passed!")

if __name__ == "__main__":
    test_fallback_formatting()
