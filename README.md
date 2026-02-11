# Sexism Detection & Mitigation System

> **Status:** Production-Ready (Modernized 2023 → 2026)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

## Overview

A production-ready REST API for detecting sexist language in text using transformer models. Originally developed as Master's NLP research at Université de Lorraine (2023), this project has been modernized into a complete ML system with API, Docker containerization, and web interface.

### Performance
- **Binary Classification:** F1 0.827, Accuracy 87.85%
- **Model:** RoBERTa-base fine-tuned on EDOS dataset
- **Inference Speed:** ~80-95ms per prediction
- **Dataset:** 20,000 labeled comments (Reddit, Gab)

---

## Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/Kira252Grudinina/etp-23-modernized.git
cd etp-23-modernized

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Run API server
python run_api.py
```

**API:** http://127.0.0.1:8000  
**Docs:** http://127.0.0.1:8000/docs

### Docker

```bash
# Start both API and web interface
docker-compose up

# Access services:
# API: http://localhost:8000
# Gradio UI: http://localhost:7860
```

---

## Usage Examples

### Python

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "Women should stay in the kitchen"}
)

result = response.json()
print(f"Label: {result['label']}")           # sexist
print(f"Confidence: {result['confidence']}")  # 0.668
```

### cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

### Batch Processing

```python
response = requests.post(
    "http://127.0.0.1:8000/predict/batch",
    json={
        "texts": [
            "Women are wonderful people",
            "Women should stay in the kitchen"
        ]
    }
)
```

---

## Project Structure

```
etp-23-modernized/
├── src/sexism_detector/        # Main Python package
│   ├── api/                    # FastAPI application & Gradio UI
│   ├── data/                   # Data processing & datasets
│   ├── models/                 # Model wrappers & training
│   └── config.py               # Configuration management
├── models/checkpoints/         # Trained model weights
├── scripts/                    # Utility scripts
├── docs/                       # Documentation
├── tests/                      # Test suite
├── Dockerfile                  # API container
├── docker-compose.yml          # Multi-service orchestration
└── run_api.py                  # API entry point
```

---

## Features

### Completed
- Modern Python 3.12 package structure
- RoBERTa-based classifier (F1: 0.827)
- REST API with FastAPI & OpenAPI docs
- Gradio web interface
- Docker containerization
- Batch prediction support
- Request validation with Pydantic
- Health checks & monitoring
- Comprehensive documentation

### API Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Single text prediction
- `POST /predict/batch` - Batch predictions (up to 100 texts)
- `GET /examples` - Example texts for testing
- `GET /docs` - Interactive API documentation

---

## Model Details

**Architecture:** RoBERTa-base + Linear classification head  
**Training:** 4 epochs, batch size 16, learning rate 2e-5  
**Device:** Trained on Tesla T4 GPU (Google Colab)  
**Categories Detected:**
- Threats and incitement
- Derogation and dehumanization
- Animosity and stereotypes
- Prejudiced discussions

---

## Development

### Testing

```bash
# Test preprocessing
python scripts/test_preprocessing.py

# Test dataset loading
python scripts/test_dataset.py

# Test API (requires server running)
python scripts/test_api.py
```

---

## Technology Stack

- **Language:** Python 3.12
- **ML Framework:** PyTorch 2.10, Transformers 5.1
- **API:** FastAPI 0.110, Uvicorn
- **UI:** Gradio 4.20
- **Containerization:** Docker, Docker Compose
- **Data:** Pandas 3.0, NumPy 1.26

---

## Credits

### Original Research (2023)
- Kira Grudinina and 8 co-authors
- Université de Lorraine, Master's NLP Program
- Original codebase: Jupyter notebooks

### Modernization (2026)
- Kira Grudinina
- Restructured to production-ready system
- Added API, Docker, and comprehensive testing

### Dataset
EDOS (Explainable Detection of Online Sexism)  
SemEval-2023 Task 10

---

## License

MIT License - See [LICENSE](LICENSE) for details

---

## Contact

**Developer:** Kira Grudinina  
**Email:** grudinina.kira@gmail.com 
**GitHub:** [@Kira252Grudinina](https://github.com/Kira252Grudinina)

---

**Last Updated:** February 2026
