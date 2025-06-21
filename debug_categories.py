#!/usr/bin/env python3
"""
Debug script to investigate the format_all_categories_for_prompt function.
"""

from risk_inference import format_all_categories_for_prompt
from risk_fine_tuner_constants import L2, MACRO_RISKS, PII_PROTECTION_CATEGORIES, PII_TYPES

print("=== Debug Categories Formatting ===")

categories = {
    'L2': L2,
    'MACRO_RISKS': MACRO_RISKS,
    'PII_PROTECTION_CATEGORIES': PII_PROTECTION_CATEGORIES,
    'PII_TYPES': PII_TYPES
}

print(f"Input categories structure:")
print(f"  L2: {type(L2)} with {len(L2)} items")
print(f"  MACRO_RISKS: {type(MACRO_RISKS)} with {len(MACRO_RISKS)} items")
print(f"  PII_PROTECTION_CATEGORIES: {type(PII_PROTECTION_CATEGORIES)} with {len(PII_PROTECTION_CATEGORIES)} items")
print(f"  PII_TYPES: {type(PII_TYPES)} with {len(PII_TYPES)} items")

print(f"\nFirst few L2 categories:")
for key, value in list(L2.items())[:3]:
    print(f"  {key}: {value}")

result = format_all_categories_for_prompt(categories)
print(f"\nResult:")
print(f"  Length: {len(result)} characters")
print(f"  Content: '{result}'")
print(f"  Contains 'Operating Model': {'Operating Model' in result}")
print(f"  Contains 'PC0': {'PC0' in result}")

print(f"\n=== Testing alternative input formats ===")

try:
    result_l2 = format_all_categories_for_prompt({'L2': L2})
    print(f"L2 only result length: {len(result_l2)}")
except Exception as e:
    print(f"L2 only failed: {e}")

try:
    alt_categories = {
        'l2_categories': L2,
        'macro_risks': MACRO_RISKS,
        'pii_categories': PII_PROTECTION_CATEGORIES,
        'pii_types': PII_TYPES
    }
    result_alt = format_all_categories_for_prompt(alt_categories)
    print(f"Alternative keys result length: {len(result_alt)}")
except Exception as e:
    print(f"Alternative keys failed: {e}")
