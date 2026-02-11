#!/usr/bin/env python
"""
Training script for binary sexism classifier.

Usage:
    python scripts/train_binary_classifier.py --epochs 4 --batch-size 16
"""

import sys
sys.path.insert(0, '.')

import torch
from torch import nn, optim
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, f1_score
import argparse
from pathlib import Path
from tqdm import tqdm

from src.sexism_detector.models.classifier import RobertaClassifier
from src.sexism_detector.data.dataset import get_binary_dataloaders
from src.sexism_detector.config import CHECKPOINTS_DIR


def train_epoch(model, data_loader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        
        # Calculate loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(actual_labels, predictions)
    f1_macro = f1_score(actual_labels, predictions, average='macro')
    
    return accuracy, f1_macro, predictions, actual_labels


def train(
    model_name: str = "roberta-base",
    num_epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    save_path: str = None,
):
    """
    Main training function.
    
    Args:
        model_name: HuggingFace model name
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        max_length: Maximum sequence length
        save_path: Path to save best model
    """
    print("=" * 60)
    print("Training Binary Sexism Classifier")
    print("=" * 60)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"\nDevice: {device}")
    print(f"Model: {model_name}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    # Load data
    print("\nLoading data...")
    train_loader, dev_loader = get_binary_dataloaders(
        batch_size=batch_size,
        max_length=max_length,
        model_name=model_name
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = RobertaClassifier(model_name, num_classes=2)
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"\nTraining Loss: {train_loss:.4f}")
        
        # Evaluate
        accuracy, f1_macro, predictions, labels = evaluate(model, dev_loader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation F1 (Macro): {f1_macro:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(
            labels,
            predictions,
            target_names=['not_sexist', 'sexist'],
            digits=4
        ))
        
        # Save best model
        if f1_macro > best_f1:
            best_f1 = f1_macro
            if save_path:
                save_file = Path(save_path)
                save_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model, save_file)
                print(f"\nBest model saved to {save_path} (F1: {best_f1:.4f})")
    
    print("\n" + "=" * 60)
    print(f"Training Complete! Best F1: {best_f1:.4f}")
    print("=" * 60)
    
    return model, best_f1


def main():
    parser = argparse.ArgumentParser(description='Train binary sexism classifier')
    parser.add_argument('--model-name', type=str, default='roberta-base',
                       help='HuggingFace model name')
    parser.add_argument('--epochs', type=int, default=4,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--save-path', type=str,
                       default=str(CHECKPOINTS_DIR / 'roberta_base_binary.pt'),
                       help='Path to save model')
    
    args = parser.parse_args()
    
    train(
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()
