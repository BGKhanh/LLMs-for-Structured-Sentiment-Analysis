import gradio as gr
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from demo.utils import convert_ssa_to_spacy

# === GLOBAL CONFIG & STATE ===
AVAILABLE_MODELS = [
    "Mock Model (Test)", 
    "google/gemma-1.1-2b-it", 
    "google/gemma-1.1-7b-it",
    "vinallama/vinallama-7b-chat"
]

AVAILABLE_PROMPTS = [
    "few_shot", 
    "zero_shot_cot", 
    "few_shot_cot", 
    "plan_and_solve"
]

# === DEMO MANAGER CLASS ===
class DemoManager:
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.history = []
        
    def load_model(self, model_name):
        """Load model (Mock or Real)."""
        if self.current_model_name == model_name and self.model is not None:
            return f"Model '{model_name}' already loaded."
        
        status = f"Loading {model_name}..."
        print(status)
        
        try:
            if "Mock" in model_name:
                # Mock Model
                self.model = "MOCK_OBJECT"
                time.sleep(1) # Simulate loading
            else:
                # --- REAL MODEL LOADING CODE (Uncomment to use) ---
                # from src.config import Config
                # from src.model import GemmaModel, QwenModel # Import others as needed
                # 
                # # Create dummy config for loading
                # config_dict = {
                #     "init_args": {
                #         "model_id": model_name,
                #         "dtype": "auto",
                #         "device_map": "auto"
                #     },
                #     "generation_args": {"enable_thinking": False}
                # }
                # self.model = GemmaModel(config_dict) # Or generic factory
                # self.model.load_model()
                pass
                
            self.current_model_name = model_name
            return f"✅ Model '{model_name}' loaded successfully!"
            
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"

    def predict(self, text, prompt_technique, temp, top_p, max_tokens):
        """Run inference."""
        if not self.model:
            return None, None, "⚠️ Please load a model first!", self.history
            
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # --- LOGIC INFERENCE ---
        if self.model == "MOCK_OBJECT":
            # Mock Output logic
            time.sleep(0.5) # Simulate inference
            raw_response = f"""
{{
    "opinions": [
        {{
            "Source": [["tôi"], ["tôi"]],
            "Target": [["sản phẩm này"], ["sản phẩm này"]],
            "Polar_expression": [["rất thích"], ["rất thích"]],
            "Polarity": "Positive",
            "Intensity": "Strong"
        }},
        {{
            "Source": [["nhân viên"], ["nhân viên"]],
            "Target": [["thái độ"], ["thái độ"]],
            "Polar_expression": [["tệ"], ["tệ"]],
            "Polarity": "Negative",
            "Intensity": "Standard"
        }}
    ]
}}
```"""
            processed_json = json.loads(raw_response.replace("", "").replace("```", ""))
            full_input_preview = f"[System Prompt ({prompt_technique})]\n...\n[User Input]\n{text}"
            
        else:
            # --- REAL INFERENCE LOGIC (Uncomment) ---
            # 1. Init Prompt Template
            # from src.prompt_templates import FewShotPrompt, ZeroShotCoTPrompt # etc
            # template = ... (init based on prompt_technique)
            # template.prepare()
            # sys_p, usr_p = template.get_prompt(text, "demo_id")
            # full_input_preview = f"{sys_p}\n\n{usr_p}"
            #
            # 2. Generate
            # self.model.config["temperature"] = temp # Update params dynamically if supported
            # raw_res, _ = self.model.generate_single(sys_p, usr_p)
            # processed_json = json.loads(self.model.extract_response(raw_res))
            pass

        exec_time = time.time() - start_time
        
        # --- VISUALIZATION ---
        html_graph = convert_ssa_to_spacy(text, processed_json)
        
        # --- UPDATE HISTORY ---
        run_data = {
            "id": len(self.history) + 1,
            "time": timestamp,
            "exec": f"{exec_time:.2f}s",
            "text": text,
            "result": processed_json
        }
        self.history.insert(0, run_data) # Add to top
        
        # Format metadata for display
        metadata = f"""
        Timestamp: {timestamp}
        Execution Time: {exec_time:.2f}s
        Params: Temp={temp}, Top-p={top_p}, MaxTokens={max_tokens}
        """
        
        return html_graph, processed_json, metadata, self.format_history()

    def format_history(self):
        """Convert history list to list of lists for Dataframe."""
        return [[r["id"], r["time"], r["exec"], r["text"][:50]+"..."] for r in self.history]

# Initialize Manager
demo_mgr = DemoManager()

# === GRADIO UI ===
with gr.Blocks(title="SSA Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Structured Sentiment Analysis (SSA) Demo")
    
    # State
    history_state = gr.State(value=[])
    
    # --- ROW 1: SETUP ---
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Model Setup")
            model_dropdown = gr.Dropdown(choices=AVAILABLE_MODELS, value=AVAILABLE_MODELS[0], label="Select Model")
            load_btn = gr.Button("Load Model", variant="primary")
            load_status = gr.Markdown("Ready to load.")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Parameters")
            prompt_dropdown = gr.Dropdown(choices=AVAILABLE_PROMPTS, value="few_shot", label="Prompt Technique")
            with gr.Accordion("Advanced Generation Params", open=False):
                temp_slider = gr.Slider(0.0, 2.0, value=0.1, label="Temperature")
                top_p_slider = gr.Slider(0.0, 1.0, value=0.9, label="Top-p")
                max_tok_slider = gr.Slider(50, 2048, value=512, label="Max Tokens")

    # --- ROW 2: INFERENCE ---
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Input")
            input_text = gr.Textbox(
                label="Text to Analyze", 
                lines=5, 
                placeholder="Nhập câu tiếng Việt cần phân tích...",
                value="Tôi rất thích sản phẩm này nhưng nhân viên thái độ tệ quá."
            )
            run_btn = gr.Button("🚀 Run Inference", variant="primary", size="lg")
            
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            with gr.Tabs():
                with gr.TabItem("🕸️ SSA Graph"):
                    html_output = gr.HTML(label="Dependency Graph")
                
                with gr.TabItem("📄 JSON Result"):
                    json_output = gr.JSON(label="Structured Output")
                    
                with gr.TabItem("ℹ️ Metadata"):
                    meta_output = gr.Textbox(label="Run Details", lines=5)

    # --- ROW 3: HISTORY ---
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🕰️ History")
            history_table = gr.Dataframe(
                headers=["ID", "Time", "Exec Time", "Input Preview"],
                datatype=["number", "str", "str", "str"],
                label="Recent Runs",
                interactive=False
            )

    # === EVENT LISTENERS ===
    
    # Load Model
    load_btn.click(
        fn=demo_mgr.load_model,
        inputs=[model_dropdown],
        outputs=[load_status]
    )
    
    # Run Inference
    run_btn.click(
        fn=demo_mgr.predict,
        inputs=[input_text, prompt_dropdown, temp_slider, top_p_slider, max_tok_slider],
        outputs=[html_output, json_output, meta_output, history_table]
    )

if __name__ == "__main__":
    demo.launch()### Hướng Dẫn Chạy Demo
