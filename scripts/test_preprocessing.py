#!/usr/bin/env python
"""Quick test of preprocessing module."""

import sys
sys.path.insert(0, '.')

from src.sexism_detector.data.preprocessing import get_preprocessor
from src.sexism_detector.config import EDOS_AGGREGATED_PATH
import pandas as pd


def main():
    print("=" * 60)
    print("Testing Preprocessing Module")
    print("=" * 60)
    
    # Load sample data
    print("\n1. Loading data...")
    df = pd.read_csv(EDOS_AGGREGATED_PATH, nrows=10)
    print(f"   ✅ Loaded {len(df)} sample rows")
    
    # Create preprocessor
    print("\n2. Creating preprocessor...")
    preprocessor = get_preprocessor("bert_optimized")
    print("   ✅ Preprocessor created")
    
    # Test on samples
    print("\n3. Testing on sample texts...")
    sample_texts = df['text'].head(3).tolist()
    
    print("\n   Original texts:")
    for i, text in enumerate(sample_texts, 1):
        print(f"   {i}. {text[:80]}...")
    
    # Process
    cleaned = preprocessor.process_batch(sample_texts)
    
    print("\n   Cld texts:")
    for i, text in enumerate(cleaned, 1):
        print(f"   {i}. {text[:80]}...")
    
    # Show cache info
    print(f"\n4. Cache info: {preprocessor.get_cache_info()}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
