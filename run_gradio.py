#!/usr/bin/env python
"""
Run the Gradio interface.

Usage:
    python run_gradio.py
"""

from src.sexism_detector.api.gradio_app import demo

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
