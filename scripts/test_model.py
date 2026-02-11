#!/usr/bin/env python
"""Test model classifier."""

import sys
sys.path.insert(0, '.')

from src.sexism_detector.models.classifier import SexismClassifier


def main():
    print("=" * 60)
    print("Testing Model Classifier")
    print("=" * 60)
    
    # Initialize model
    print("\n1. Initializing binary classifier...")
    classifier = SexismClassifier(
        model_name="roberta-base",
        num_classes=2
    )
    print("   [OK] Model initialized")
    
    # Test predictions
    print("\n2. Testing predictions...")
    test_texts = [
        "Women are wonderful people",
        "Women should stay in the kitchen",
        "I respect everyone regardless of gender"
    ]
    
    for text in test_texts:
        result = classifier.predict(text)
        print(f"\n   Text: {text[:60]}...")
        print(f"   Prediction: {result['label']}")
        print(f"   Confidence: {result['confidence']:.3f}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Model wrapper working!")
    print("=" * 60)
    print("\nNote: These are predictions from an untrained model.")
    print("Training code will be added in Day 4.")


if __name__ == "__main__":
    main()

