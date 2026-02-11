
# Sexism Detection & Mitigation System

> **Status:** Modernization in Progress (2023 to 2025)

## Project Overview

Originally developed as Master's NLP research at Université de Lorraine (2023), this system detects and mitigates sexist language in social media using transformer models. Currently modernizing to production standards.

### Original Results (2023)
- **Binary Classification:** F1 0.83 (RoBERTa-large)
- **Multi-class Classification:** F1 0.66 (4 severity categories)
- **Text Mitigation:** BLEU 0.17 (BART-large)

### Dataset
- **EDOS** (Explainable Detection of Online Sexism)
- 20,000 labeled comments from Reddit and Gab
- Categories: Threat, Derogation, Animosity, Prejudiced Discussion

---

## Quick Start

### Installation
```bash
# Clone repository
git clone [your-repo-url]
cd etp-23

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

### Run the API
```bash
# Start the API server
python run_api.py

# API will be available at: http://127.0.0.1:8000
# Interactive docs at: http://127.0.0.1:8000/docs
```

### Quick Test
```bash
# In a new terminal
python scripts/test_api.py
```

### Use the API
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "Women should stay in the kitchen"}
)

result = response.json()
print(f"Label: {result['label']}")        # sexist
print(f"Confidence: {result['confidence']}")  # 0.668
```
---

## Project Structure
```
etp-23/
├── src/sexism_detector/      # Maage
│   ├── data/                  # Data processing
│   │   └── preprocessing.py   # Text cleaning module
│   ├── models/                # Model implementations
│   ├── training/              # Training scripts
│   ├── evaluation/            # Metrics & evaluation
│   └── api/                   # REST API (coming soon)
├── data/
│   └── raw/                   # EDOS datasets
├── notebooks/
│   └── archive_2023/          # Original research notebooks
├── scripts/                   # Utility scripts
└─
? [x] Modern Python package structure
- [x] Dependencies: transformers 5.1, torch 2.10, pandas 3.0
- [x] Configuration management system
- [x] Text preprocessing module with caching
- [x] Data organization and loading
- [x] Git workflow on feature branch

## In Progress 
- [ ] Model wrappers (RoBERTa, DeBERTa, BART)
- [ ] Training scripts with experiment tracking
- [ ] FastAPI REST API
- [ ] Unit tests & CI/CD
- [ ] Docker containerization
- [ ] Live demo deployment

---

## Testing
```bash
# Test preprocessing
python scripts/test_preprocessing.py
```

---

## Development

**Current branch:** `modernization-2025`  
**Python version:** 3.12+  
**Key technologies:** Transformers, PyTorch, FastAPI (planned)

### Next Steps
1. Refactor model code from notebooks
2. Create training pipeline
3. Build REST API
4. Add comprehensive tests

---

## Team

**Original Research (2023):**
- Kira Grudinina and 8 co-authors
- Université de Lorraine, Master's NLP

**Modernization (2025):**
- Kira Grudinina

---

## License

IT License - See LICENSE for details

---

**Last Updated:** February 2025
