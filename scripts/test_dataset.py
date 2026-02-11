#!/usr/bin/env python
"""Test dataset loader."""

import sys
sys.path.insert(0, '.')

from src.sexism_detector.data.dataset import (
    load_edos_data,
    prepare_binary_data,
    get_binary_dataloaders
)


def main():
    print("=" * 60)
    print("Testing Dataset Loader")
    print("=" * 60)
    
    # Test 1: Load raw data
    print("\n1. Loading EDOS data...")
    train_df, dev_df, test_df = load_edos_data(task="binary")
    print("   [OK] Data loaded successfully")
    
    # Test 2: Prepare binary data
    print("\n2. Preparing binary data...")
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = \
        prepare_binary_data(train_df, dev_df, test_df)
    
    print(f"   Train samples: {len(train_texts)}")
    print(f"   Dev samples: {len(dev_texts)}")
    print(f"   Test samples: {len(test_texts)}")
    
    # Show sample
    print(f"\n   Sample text: {train_texts[0][:80]}...")
    print(f"   Sample label: {train_labels[0]} ({'sexist' if train_labels[0] == 1 else 'not_sexist'})")
    
    # Test 3: Create dataloaders
    print("\n3. Creating dataloaders...")
    train_loader, dev_loader = get_binary_dataloaders(batch_size=8)
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Dev batches: {len(dev_loader)}")
    
    # Test 4: Get one batch
    print("\n4. Testing batch retrieval...")
    batch = next(iter(train_loader))
    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   Input IDs shape: {batch['input_ids'].shape}")
    print(f"   Attention mask shape: {batch['attention_mask'].shape}")
    print(f"   Labels shape: {batch['labels'].shape}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Dataset loader working!")
    print("=" * 60)


if __name__ == "__main__":
    main()
