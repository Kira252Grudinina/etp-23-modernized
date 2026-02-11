#!/usr/bin/env python
"""
Test the FastAPI endpoints.

Run this after starting the API server with: python run_api.py
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"


def test_health():
    """Test health endpoint."""
    print("\n1. Testing Health Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_single_prediction():
    """Test single prediction endpoint."""
    print("\n2. Testing Single Prediction")
    print("=" * 60)
    
    test_texts = [
        "Women should stay in the kitchen",
        "Women are wonderful people",
        "Everyone deserves equal opportunities"
    ]
    
    for text in test_texts:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text, "return_probabilities": True}
        )
        result = response.json()
        
        print(f"Text: {text}")
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
        print()


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\n3. Testing Batch Prediction")
    print("=" * 60)
    
    texts = [
        "Women are wonderful people",
        "Women should stay in the kitchen",
        "Girls are too emotional for leadership",
        "Everyone deserves respect regardless of gender"
    ]
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json={"texts": texts, "return_probabilities": True}
    )
    result = response.json()
    
    print(f"Total texts: {result['count']}")
    print(f"Total processing time: {result['total_processing_time_ms']:.2f}ms")
    print(f"Average time per text: {result['total_processing_time_ms']/result['count']:.2f}ms\n")
    
    for pred in result['predictions']:
        print(f"Text: {pred['text'][:50]}...")
        print(f"Label: {pred['label']} (confidence: {pred['confidence']:.3f})")
        print()


def test_examples():
    """Test examples endpoint."""
    print("\n4. Testing Examples Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/examples")
    examples = response.json()
    
    print("Sexist examples:")
    for ex in examples['sexist_examples']:
        print(f"  - {ex}")
    
    print("\nNot sexist examples:")
    for ex in examples['not_sexist_examples']:
        print(f"  - {ex}")
    print()


def main():
    print("=" * 60)
    print("Sexism Detection API - Test Suite")
    print("=" * 60)
    
    try:
        # Check if API is running
        response = requests.get(BASE_URL)
        print(f"\nAPI Status: Running on {BASE_URL}")
        print(f"Version: {response.json()['version']}\n")
        
        # Run tests
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_examples()
        
        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print(f"\nError: Could not connect to API at {BASE_URL}")
        print("Please start the API server first:")
        print("  python run_api.py\n")


if __name__ == "__main__":
    main()

