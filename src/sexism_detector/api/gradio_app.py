"""
Gradio web interface for sexism detection.

Provides a user-friendly UI for testing the model.
"""

import gradio as gr
import requests
import os
from typing import Tuple


# API endpoint
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def predict_text(text: str) -> Tuple[str, str, str]:
    """
    Predict sexism for input text.
    
    Args:
        text: Input text to classify
        
    Returns:
        Tuple of (label, confidence, probabilities_html)
    """
    if not text.strip():
        return "Error", "Please enter some text", ""
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text, "return_probabilities": True},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            label = result['label']
            confidence = result['confidence']
            probs = result.get('probabilities', [])
            
            # Format label with emoji
            label_display = f"{'🚫 SEXIST' if label == 'sexist' else '✅ NOT SEXIST'}"
            
            # Format confidence
            confidence_display = f"{confidence:.1%}"
            
            # Create probability bars
            # Create probability bars
            if probs:
                prob_html = f"""
                <div style="margin-top: 10px;">
                    <div style="margin-bottom: 5px;">
                        <span style="display: inline-block; width: 100px;">Not Sexist:</span>
                        <div style="display: inline-block; width: 200px; background: #e0e0e0; border-radius: 5px;">
                            <div style="width: {probs[0]*100}%; background: #4CAF50; height: 20px; border-radius: 5px;"></div>
                        </div>
                        <span style="margin-left: 10px;">{probs[0]:.1%}</span>
                    </div>
                    <div>
                        <span style="display: inline-block; width: 100px;">Sexist:</span>
                        <div style="display: inline-block; width: 200px; background: #e0e0e0; border-radius: 5px;">
                            <div style="width: {probs[1]*100}%; background: #f44336; height: 20px; border-radius: 5px;"></div>
                        </div>
                        <span style="margin-left: 10px;">{probs[1]:.1%}</span>
                    </div>
                </div>
                """
            else:
                prob_html = ""
            return label_display, confidence_display, prob_html
        else:
            return "Error", f"API Error: {response.status_code}", ""
    
    except requests.exceptions.ConnectionError:
        return "Error", "Cannot connect to API. Is it running?", ""
    except Exception as e:
        return "Error", f"Error: {str(e)}", ""


# Example texts
examples = [
    ["Women are wonderful people"],
    ["Women should stay in the kitchen"],
    ["Girls are too emotional for leadership"],
    ["Everyone deserves equal opportunities"],
    ["She's just a pretty face"],
    ["Respect people regardless of gender"]
]


# Create Gradio interface
with gr.Blocks(title="Sexism Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🛡️ Sexism Detection System
        
        This system uses a fine-tuned RoBERodel to detect sexist language in text.
        
        **Model Performance:** F1 Score 0.827 | Accuracy 87.85%
        
        **Dataset:** EDOS (Explainable Detection of Online Sexism) - 20,000 samples
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter text to analyze",
                placeholder="Type or paste text here...",
                lines=5
            )
            
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column(scale=1):
            label_output = gr.Textbox(label="Prediction", interactive=False)
            confidence_output = gr.Textbox(label="Confidence", interactive=False)
            probs_output = gr.HTML(label="Probabilities")
    
    gr.Markdown("### Try these examples:")
    gr.Examples(
        examples=examples,
        inputs=text_input,
        label="Example Texts"
    )
    
    gr.Markdown(
        """
        ---
        ### About
        
        This model was trained to detect sexist language across four categories:
        - **Threats** - Threats, plans to harm, and incitement
        - **Derogation** - Dehumanizing attacks and objectification
        - **Animosity** - Implicit sexism and stereotypes
        - **Prejudiced** - Denial of discrimination
        
        **Note:** This is a binary classifier showing sexist vs. not sexist.
        
        **Technology:** RoBERTa-base, PyTorch, FastAPI, Gradio
        
        **Author:** Kira Grudinina | **Year:** 2025
        """
    )
    
    # Event handlers
    submit_btn.click(
        fn=predict_text,
        inputs=text_input,
        outputs=[label_output, confidence_output, probs_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", "", ""),
        inputs=None,
        outputs=[text_input, label_output, confidence_output, probs_output]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
