# Model Checkpoints

## Trained Models

### Binary Classifier
- **File:** `roberta_base_binary.pt`
- **Size:** 476 MB
- **Performance:** F1 0.827, Accuracy 87.85%
- **Training:** Google Colab (Tesla T4 GPU)
- **Dataset:** EDOS (14,000 train, 2,000 validation)
- **Architecture:** RoBERTa-base with classification head
- **Epochs:** 4
- **Hyperparameters:**
  - Batch size: 16
  - Learning rate: 2e-5
  - Max length: 512

## How to Get the Model

### Option 1: Train Yourself
Use the provided Colab notebook:
1. Upload `Train_Sexism_Classifier_Colab.ipynb` to Google Colab
2. Enable GPU (Runtime → Change runtime type → GPU)
3. Upload EDOS dataset
4. Run all cells (~90 minutes)
5. Download trained model

### Option 2: Use Pre-trained
Contact the repository owner for access to the trained model.

## Loading the Model
```python
from src.sexism_detector.models.classifier import SexismClassifier

classifier = SexismClassifier.from_pretrained(
    'models/checkpoints/roberta_base_binary.pt',
    num_classes=2
)

resu classifier.predict("Your text here")
print(result['label'], result['confidence'])
```

## Model Performance

| Metric | Score |
|--------|-------|
| F1 Macro | 0.827 |
| Accuracy | 87.85% |
| Precision (not_sexist) | 90.3% |
| Recall (not_sexist) | 93.4% |
| Precision (sexist) | 77.2% |
| Recall (sexist) | 69.6% |

## Training Details

Trained using:
- Optimizer: AdamW
- Scheduler: Linear with warmup (500 steps)
- Loss: CrossEntropyLoss
- Device: CUDA (Colab T4 GPU)
- Training time: ~90 minutes

## Notes

- Model file is excluded from Git due to size (476 MB)
- Use Git LFS for version control of large model files
- For production, consider hosting on Hugging Face Hub or AWS S3
