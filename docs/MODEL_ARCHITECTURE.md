# Model Architecture Documentation

## Overview

This project implements sexism detection using RoBERTa-based classifiers with two separate tasks:
1. Binary classification (sexist vs. not sexist)
2. Multi-class classification (4 severity categories)

---

## Architecture

### Base Model: RoBERTa
```
Input Text
    ↓
RoBERTa Tokenizer (max_length=512)
    ↓
RoBERTa Encoder (roberta-base or roberta-large)
    ↓
Pooler Output (768-dim for base, 1024-dim for large)
    ↓
Dropout (p=0.1)
    ↓
Linear Classification Head
    ↓
Output Logits (2 classes or 4 classes)
```

### Model Specifications

**Binary Classifier:**
- Input: Text sequences (max 512 tokens)
- Architecture: RoBERTa-base + Linear head
- Output: 2 classes [not_sexist, sexist]
- Loss: CrossEntropyLoss
- Original Performance: F1 0.83

**Multi-class Classifier:**
- Input: Text sequences (sexist comments only)
- Architecture: RoBERTa-base + Linear head
- Output: 4 classes [threat, derogation, animosity, prejudiced]
- Loss: CrossEntropyLoss
erformance: F1 0.66

---

## Training Configuration

### Hyperparameters (from original notebook)
```python
model_name = "roberta-base"
max_length = 512
batch_size = 16
num_epochs = 4
learning_rate = 2e-5
warmup_steps = 500
dropout = 0.1
```

### Optimizer
- AdamW optimizer
- Linear learning rate schedule with warmup

### Data Split
- Train: 14,000 samples (binary) / 3,398 samples (multiclass)
- Dev: 2,000 samples (binary) / ~500 samples (multiclass)
- Test: 4,000 samples (binary) / ~1,000 samples (multiclass)

---

## Class Implementation

### RobertaClassifier (PyTorch Module)
```python
class RobertaClassifier(nn.Module):
    def __init__(self, roberta_model_name, num_classes, dropout=0.1):
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_classes)
```

### SexismClassifier (High-level Wrapper)

Provides:
- Easy initialization with auto device detection
- Prediction interface with probabilities
- Model save/load functionality
- Support for both binary and multi-class

---

## Category Definitions

### Binary Classes
0. **not_sexist** - No sexist content detected
1. **sexist** - Sexist content detected

### Multi-class Categories (Severity)
0. **threat** - Threats, plans to harm and incitement
1. **derogation** - Dehumanizing attacks, objectification
2. **animosity** - Implicit sexism, stereotypes
3. **prejudiced** - Denial of discrimination

---

## Preprocessing Pipeline

1. Remove emoji (converted to text or removed)
2. Remove [URL] and [USER] placeholders
3. Normalize whitespace
4. Tokenize with RoBERTa tokenizer
5. Pad/truncate to max_length=512

---

## Performance Targets

Based on original 2023 research:

| Task | Model | Metric | Score |
|------|-------|--------|-------|
| Binary | RoBERTa-base | F1 Macro | 0.83 |
| Binary | RoBERTa-base | Accuracy | ~0.85 |
| Multi-class | RoBERTa-base | F1 Macro | 0.66 |
| Multi-class | RoBERTa-base | Accuracy | ~0.70 |

---

## Inference Example
```python
from src.sexism_detector.models.classifier import SexismClassifier

# Load model
classifier = SexismClassifier(
    model_name="roberta-base",
    num_classes=2
)

# Predict
result = classifier.predict("Women should stay in the kitchen")
print(result['label'])  # 'sexist' or 'not_sexist'
print(result['confidence'])  # 0.0 to 1.0
```

---

## Model Files

Trained models are saved in: `models/checkpoints/`

- `roberta_base_binary.pt` - Binary classifier
- `roberta_base_multiclass.pt` - Multi-class classifier
- `roberta_large_binary.pt` - Large model (optional)

---

## Future Improvements

Potential enhancements:
1. Use LoRA for parameter-efficient fine-tuning
2. Implement 4-bit quantization for faster inference
3. Add ensemble methods
4. Experiment with larger models (roberta-large, deberta)
5. Add uncertainty estimation
6. Implement active learning pipeline

---

## References

- Original Paper: "Ethics Therapy Project: Detection and Text Style Transfer of Sexist Language"
- EDOS Dataset: SemEval-2023 Task 10
- RoBERTa: Liu et al., 2019
