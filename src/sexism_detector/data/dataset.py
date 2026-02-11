"""
Dataset loaders for EDOS (Explainable Detection of Online Sexism).

Handles both binary and multi-class classification tasks.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from ..config import EDOS_AGGREGATED_PATH, CATEGORY_LABELS


class SexismDataset(Dataset):
    """PyTorch Dataset for sexism detection."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: RobertaTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of text samples
            labels: List of integer labels
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_edos_data(
    task: str = "binary",
    text_column: str = "text",
    preprocessed: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load EDOS dataset and split into train/dev/test.
    
    Args:
        task: Either "binary" or "multiclass"
        text_column: Which text column to use
        preprocessed: If True, use preprocessed text columns
        
    Returns:
        Tuple of (train_df, dev_df, test_df)
    """
    # Load data
    df = pd.read_csv(EDOS_AGGREGATED_PATH)
    
    # Use preprocessed column if requested
    if preprocessed and 'text_en_emoji_rm_url' in df.columns:
        text_column = 'text_en_emoji_rm_url'
    
    # Split by the existing split column
    train_df = df[df['split'] == 'train'].copy()
    dev_df = df[df['split'] == 'dev'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Dev:   {len(dev_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    if task == "binary":
        print(f"\nBinary task distribution (train):")
        print(train_df['label_sexist'].value_counts())
    else:
        # Filter to only sexist samples for multiclass
        train_df = train_df[train_df['label_sexist'] == 'sexist'].copy()
        dev_df = dev_df[dev_df['label_sexist'] == 'sexist'].copy()
        test_df = test_df[test_df['label_sexist'] == 'sexist'].copy()
        
        print(f"\nMulticlass task (sexist only):")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Dev:   {len(dev_df)} samples")
        print(f"  Test:  {len(test_df)} samples")
        print(f"\nCategory distribution (train):")
        print(train_df['label_category'].value_counts())
    
    return train_df, dev_df, test_df


def prepare_binary_data(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "text",
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Prepare data for binary classification.
    
    Args:
        train_df, dev_df, test_df: DataFrames from load_edos_data
        text_column: Column containing text
        
    Returns:
        Tuple of (train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels)
    """
    # Extract texts
    train_texts = train_df[text_column].tolist()
    dev_texts = dev_df[text_column].tolist()
    test_texts = test_df[text_column].tolist()
    
    # Convert labels to binary (0: not_sexist, 1: sexist)
    train_labels = [1 if label == "sexist" else 0 for label in train_df['label_sexist'].tolist()]
    dev_labels = [1 if label == "sexist" else 0 for label in dev_df['label_sexist'].tolist()]
    test_labels = [1 if label == "sexist" else 0 for label in test_df['label_sexist'].tolist()]
    
    return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels


def prepare_multiclass_data(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "text",
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Prepare data for multi-class classification (4 categories).
    
    Args:
        train_df, dev_df, test_df: DataFrames from load_edos_data (sexist only)
        text_column: Column containing text
        
    Returns:
        Tuple of (train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels)
    """
    # Category mapping from original notebook
    category_mapping = {
        '1. threats, plans to harm and incitement': 0,
        '2. derogation': 1,
        '3. animosity': 2,
        '4. prejudiced discussions': 3
    }
    
    # Extract texts
    train_texts = train_df[text_column].tolist()
    dev_texts = dev_df[text_column].tolist()
    test_texts = test_df[text_column].tolist()
    
    # Map categories to integers
    train_labels = [category_mapping[cat] for cat in train_df['label_category'].tolist()]
    dev_labels = [category_mapping[cat] for cat in dev_df['label_category'].tolist()]
    test_labels = [category_mapping[cat] for cat in test_df['label_category'].tolist()]
    
    return train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels


def create_data_loaders(
    train_texts: List[str],
    train_labels: List[int],
    dev_texts: List[str],
    dev_labels: List[int],
    tokenizer: RobertaTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        train_texts, train_labels: Training data
        dev_texts, dev_labels: Validation data
        tokenizer: Tokenizer instance
        batch_size: Batch size for training
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_loader, dev_loader)
    """
    train_dataset = SexismDataset(train_texts, train_labels, tokenizer, max_length)
    dev_dataset = SexismDataset(dev_texts, dev_labels, tokenizer, max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Mac compatibility
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, dev_loader


def get_binary_dataloaders(
    batch_size: int = 16,
    max_length: int = 512,
    model_name: str = "roberta-base",
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to get binary classification dataloaders.
    
    Args:
        batch_size: Batch size
        max_length: Maximum sequence length
        model_name: Model name for tokenizer
        
    Returns:
        Tuple of (train_loader, dev_loader)
    """
    # Load data
    train_df, dev_df, test_df = load_edos_data(task="binary")
    
    # Prepare data
    train_texts, train_labels, dev_texts, dev_labels, _, _ = prepare_binary_data(
        train_df, dev_df, test_df
    )
    
    # Create tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    # Create dataloaders
    return create_data_loaders(
        train_texts, train_labels,
        dev_texts, dev_labels,
        tokenizer, batch_size, max_length
    )


def get_multiclass_dataloaders(
    batch_size: int = 16,
    max_length: int = 512,
    model_name: str = "roberta-base",
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to get multi-class classification dataloaders.
    
    Args:
        batch_size: Batch size
        max_length: Maximum sequence length
        model_name: Model name for tokenizer
        
    Returns:
        Tuple of (train_loader, dev_loader)
    """
    # Load data (will filter to sexist only)
    train_df, dev_df, test_df = load_edos_data(task="multiclass")
    
    # Prepare data
    train_texts, train_labels, dev_texts, dev_labels, _, _ = prepare_multiclass_data(
        train_df, dev_df, test_df
    )
    
    # Create tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    # Create dataloaders
    return create_data_loaders(
        train_texts, train_labels,
        dev_texts, dev_labels,
        tokenizer, batch_size, max_length
    )
