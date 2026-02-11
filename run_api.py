#!/usr/bin/env python
"""
Run the FastAPI server.

Usage:
    python run_api.py
    python run_api.py --host 0.0.0.0 --port 8000
"""

import uvicorn
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Sexism Detection API')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    uvicorn.run(
        "src.sexism_detector.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
