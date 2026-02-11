"""
RoBERTa-based classifier for sexism detection.

Modernized from 2023 notebook implementation with:
- Hugging Face Trainer API (cleaner than custom loops)
- Type hints and documentation
- Flexible configuration
- Easy inference interface
"""

import torch
from torch import nn
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from typing import Dict, List, Optional, Union
import numpy as np
from pathlib import Path


class RobertaClassifier(nn.Module):
    """Custom RoBERTa classifier (from original notebook)."""
    
    def __init__(self, roberta_model_name: str, num_classes: int, dropout: float = 0.1):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.roberta.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        logits = self.linear(dropout_output)
        return logits


class SexismClassifier:
    """
    High-level wrapper for sexism classification.
    
    Supports both binary and multi-class classification.
    """
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        num_classes: int = 2,
        device: Optional[str] = None,
    ):
        """
        Initialize classifier.
        
        Args:
            model_name: HuggingFace model name (e.g., 'roberta-base', 'roberta-large')
            num_classes: Number of classes (2 for binary, 4 for multi-class)
            device: Device to use ('cuda', 'mps', 'cpu'). Auto-detected if None.
        """
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Initialize model
        self.model = RobertaClassifier(model_name, num_classes)
        self.model.to(self.device)
        
        print(f"Model loaded on device: {self.device}")
    
    def predict(
        self,
        texts: Union[str, List[str]],
        return_probs: bool = True,
        max_length: int = 512,
    ) -> Union[Dict, List[Dict]]:
        """
        Predict sexism for text(s).
        
        Args:
            texts: Single text or list of texts
            return_probs: If True, return probabilities for all classes
            max_length: Maximum sequence length
            
        Returns:
            Dictionary or list of dictionaries with predictions
        """
        # Handle single text
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=max_length,
                    padding='max_length',
                    truncation=True
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Get predictions
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                predicted_class = int(np.argmax(probs))
                
                result = {
                    'text': text,
                    'predicted_class': predicted_class,
                    'confidence': float(probs[predicted_class]),
                }
                
                if return_probs:
                    result['probabilities'] = probs.tolist()
                
                # Add label interpretation
                if self.num_classes == 2:
                    result['label'] = 'sexist' if predicted_class == 1 else 'not_sexist'
                else:
                    categories = ['threat', 'derogation', 'animosity', 'prejudiced']
                    result['category'] = categories[predicted_class]
                
                results.append(result)
        
        return results[0] if single_input else results
    
    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path], num_classes: int = 2):
        """
        Load a saved model.
        
        Args:
            model_path: Path to saved model (.pt file)
            num_classes: Number of classes (2 or 4)
            
        Returns:
            Loaded classifier instance
        """
        classifier = cls(num_classes=num_classes)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=classifier.device)
        
        # Handle different save formats
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            classifier.model.load_state_dict(state_dict['model_state_dict'])
        else:
            # Assume entire model was saved
            classifier.model = state_dict
            classifier.model.to(classifier.device)
        
        classifier.model.eval()
        return classifier
    
    def save(self, save_path: Union[str, Path]):
        """
        Save model to disk.
        
        Args:
            save_path: Path to save model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
        }, save_path)
        
        print(f"Model saved to {save_path}")


def load_binary_classifier(model_path: Optional[str] = None) -> SexismClassifier:
    """
    Convenience function to load binary classifier.
    
    Args:
        model_path: Path to saved model. If None, returns fresh model.
        
    Returns:
        Binary sexism classifier
    """
    if model_path:
        return SexismClassifier.from_pretrained(model_path, num_classes=2)
    else:
        return SexismClassifier(model_name="roberta-base", num_classes=2)


def load_multiclass_classifier(model_path: Optional[str] = None) -> SexismClassifier:
    """
    Convenience function to load multi-class classifier.
    
    Args:
        model_path: Path to saved model. If None, returns fresh model.
        
    Returns:
        Multi-class sexism classifier (4 categories)
    """
    if model_path:
        return SexismClassifier.from_pretrained(model_path, num_classes=4)
    else:
        return SexismClassifier(model_name="roberta-base", num_classes=4)
