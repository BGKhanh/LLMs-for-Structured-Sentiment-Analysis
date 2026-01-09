# app.py

import gradio as gr
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from src.demo.utils import convert_ssa_to_spacy
from src.model.HFModel import HFModel
from src.utils.postprocessing import postprocess_response
from src.utils.random_seed import setup_reproducible_environment

# Import prompt templates
from src.prompt_templates import *
# === GLOBAL CONFIG & STATE ===
AVAILABLE_MODELS = [
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

AVAILABLE_PROMPTS = {
    "few_shot": {
        "name": "Few-shot",
        "requires_pool": True,
        "params": ["n_shot", "examples_pool"]
    },
    "few_shot_cot": {
        "name": "Few-shot CoT",
        "requires_pool": False,  # Uses hardcoded examples
        "params": ["n_shot"]
    },
    "plan_and_solve": {
        "name": "Plan-and-Solve",
        "requires_pool": False,
        "params": ["plus", "n_shot"]  # n_shot only if plus=True
    },
    "re_reading": {
        "name": "Re-reading",
        "requires_pool": "conditional",  # Depends on add_method
        "params": ["add_method", "n_shot", "examples_pool"]
    },
    "re2_pas_cot": {
        "name": "RE2+PaS+CoT",
        "requires_pool": False,  # Uses hardcoded examples
        "params": ["n_shot"]
    }
}

EXAMPLES_POOL_OPTIONS = {
    "data/train.json": "Train Set",
    "data/dev.json": "Dev Set",
    "data/test.json": "Test Set"
}

# === DEMO MANAGER CLASS ===
class DemoManager:
    def __init__(self, seed=42):
        self.seed = seed
        self.setup_seed(seed)
        
        self.model = None
        self.current_model_name = None
        self.prompt_template = None
        self.current_prompt_config = None
        self.history = []
    
    def setup_seed(self, seed):
        """Setup reproducible environment with given seed."""
        try:
            setup_reproducible_environment(seed=seed)
            self.seed = seed
            print(f"✅ Reproducible environment initialized with seed={seed}")
            return f"✅ Random seed set to {seed}"
        except Exception as e:
            error_msg = f"⚠️ Warning: Could not set deterministic mode: {str(e)}"
            print(error_msg)
            return error_msg
            
    def load_model(self, model_name):
        """Load HFModel."""
        if self.current_model_name == model_name and self.model is not None:
            return f"✅ Model '{model_name}' already loaded."
        
        status = f"🔄 Loading {model_name}..."
        print(status)
        
        try:
            config = {
                "model_id": model_name,
                "init_args": {
                    "model_id": model_name,
                    "dtype": "auto",
                    "device_map": "auto",
                    "trust_remote_code": True
                },
                "generation_args": {
                    "max_new_tokens": 1024,
                    "do_sample": False,
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }
            
            self.model = HFModel(config)
            self.model.load_model()
            
            self.current_model_name = model_name
            return f"✅ Model '{model_name}' loaded successfully!"
            
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    
    def load_prompt_template(self, prompt_type, n_shot, examples_pool_path, 
                            plus_mode, re_add_method):
        """Initialize prompt template based on configuration."""
        config_key = f"{prompt_type}_{n_shot}_{examples_pool_path}_{plus_mode}_{re_add_method}"
        
        if self.current_prompt_config == config_key and self.prompt_template is not None:
            return f"✅ Prompt already loaded with same config."
        
        try:
            use_english = False  # Vietnamese for demo
            
            if prompt_type == "few_shot":
                if n_shot > 0 and not Path(examples_pool_path).exists():
                    return f"⚠️ Examples pool not found: {examples_pool_path}"
                
                self.prompt_template = FewShotPrompt(
                    eng=use_english,
                    n_shot=n_shot,
                    examples_pool_path=examples_pool_path if n_shot > 0 else None
                )
                
            elif prompt_type == "few_shot_cot":
                self.prompt_template = FewShotCoTPrompt(
                    eng=use_english,
                    n_shot=n_shot
                )
                
            elif prompt_type == "plan_and_solve":
                self.prompt_template = PlanAndSolvePrompt(
                    eng=use_english,
                    plus=plus_mode,
                    n_shot=n_shot if plus_mode else 0
                )
                
            elif prompt_type == "re_reading":
                # Check if examples pool is needed
                needs_pool = re_add_method in ["FewShot"]
                if needs_pool and n_shot > 0 and not Path(examples_pool_path).exists():
                    return f"⚠️ Examples pool not found: {examples_pool_path}"
                
                self.prompt_template = ReReadingPrompt(
                    eng=use_english,
                    add_method=re_add_method,
                    n_shot=n_shot,
                    examples_pool_path=examples_pool_path if needs_pool and n_shot > 0 else None
                )
                
            elif prompt_type == "re2_pas_cot":
                self.prompt_template = Re2PaSCoTPrompt(
                    eng=use_english,
                    n_shot=n_shot
                )
            
            # Prepare template
            self.prompt_template.prepare()
            
            self.current_prompt_config = config_key
            return f"✅ Prompt '{AVAILABLE_PROMPTS[prompt_type]['name']}' loaded!"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error loading prompt: {str(e)}"

    def predict(self, text, prompt_type, n_shot, examples_pool_path,
                plus_mode, re_add_method, temp, top_p, max_tokens):
        """Run inference with configured prompt template."""
        if not self.model:
            return None, None, "⚠️ Please load a model first!", self.format_history()
        
        # Load prompt template
        load_msg = self.load_prompt_template(
            prompt_type, n_shot, examples_pool_path, plus_mode, re_add_method
        )
        if "Error" in load_msg or "⚠️" in load_msg:
            return None, None, load_msg, self.format_history()
            
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # === GET PROMPTS FROM TEMPLATE ===
            system_prompt, user_prompt = self.prompt_template.get_prompt(
                text=text,
                sent_id=f"demo_{len(self.history) + 1}"
            )
            
            # === GENERATE ===
            raw_response, gen_time = self.model.generate_single(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temp,
                top_p=top_p,
                max_new_tokens=max_tokens
            )
            
            # === EXTRACT & POSTPROCESS ===
            json_str = self.model.extract_response(raw_response)
            processed = postprocess_response(
                response_text=json_str,
                original_text=text,
                sent_id=f"demo_{len(self.history) + 1}"
            )
            processed_json = json.loads(processed)
            
            exec_time = time.time() - start_time
            
            # === VISUALIZATION ===
            html_graph = convert_ssa_to_spacy(text, processed_json)
            
            # === COUNT TOKENS ===
            input_tokens = len(self.model.tokenizer.encode(system_prompt + user_prompt))
            output_tokens = len(self.model.tokenizer.encode(raw_response))
            
            # === BUILD PROMPT CONFIG STRING ===
            config_str = f"{AVAILABLE_PROMPTS[prompt_type]['name']}"
            if n_shot > 0:
                config_str += f" (n_shot={n_shot})"
            if prompt_type == "plan_and_solve":
                config_str += f" ({'PS+' if plus_mode else 'PS'})"
            if prompt_type == "re_reading":
                config_str += f" (method={re_add_method})"
            
            # === UPDATE HISTORY ===
            run_data = {
                "id": len(self.history) + 1,
                "time": timestamp,
                "exec": f"{exec_time:.2f}s",
                "text": text,
                "prompt_config": config_str,
                "raw_response": raw_response,
                "result": processed_json,
                "params": {
                    "temperature": temp,
                    "top_p": top_p,
                    "max_tokens": max_tokens
                },
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens
                }
            }
            self.history.insert(0, run_data)
            
            # Format metadata
            metadata = f"""⏱️  Timestamp: {timestamp}
⚡ Execution Time: {exec_time:.2f}s (Generation: {gen_time:.2f}s)
📊 Tokens: Input={input_tokens}, Output={output_tokens}
🎯 Prompt: {config_str}
🎛️  Params: Temp={temp}, Top-p={top_p}, MaxTokens={max_tokens}

📝 System Prompt Preview:
{system_prompt[:300]}...

💬 User Prompt Preview:
{user_prompt[:300]}...

🤖 Raw Response Preview:
{raw_response[:400]}{"..." if len(raw_response) > 400 else ""}
"""
            
            return html_graph, processed_json, metadata, self.format_history()
            
        except Exception as e:
            error_msg = f"❌ Error during inference: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, None, error_msg, self.format_history()

    def format_history(self):
        """Convert history to dataframe format."""
        if not self.history:
            return []
        return [
            [
                r["id"], 
                r["time"], 
                r.get("prompt_config", "N/A"),
                r["exec"], 
                r["text"][:35] + ("..." if len(r["text"]) > 35 else "")
            ] 
            for r in self.history
        ]

# Initialize Manager
demo_mgr = DemoManager()

# === GRADIO UI ===
with gr.Blocks(title="SSA Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Structured Sentiment Analysis (SSA) Demo")
    gr.Markdown("### Powered by HFModel + Prompt Engineering Techniques")
    
    # --- MODEL SETUP ---
    with gr.Row(variant="panel"):
        with gr.Column():
            gr.Markdown("### ⚙️ Model Setup")
            model_dropdown = gr.Dropdown(
                choices=AVAILABLE_MODELS, 
                value=AVAILABLE_MODELS[0], 
                label="Select Model"
            )
            load_model_btn = gr.Button("🔄 Load Model", variant="primary")
            model_status = gr.Markdown("_Ready to load..._")

    # --- PROMPT CONFIGURATION ---
    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            gr.Markdown("### 🎯 Prompt Configuration")
            prompt_dropdown = gr.Dropdown(
                choices=list(AVAILABLE_PROMPTS.keys()),
                value="few_shot",
                label="Prompt Technique"
            )
            
            with gr.Row():
                n_shot_slider = gr.Slider(0, 5, value=3, step=1, label="N-shot", info="Number of examples")
                examples_pool_dropdown = gr.Dropdown(
                    choices=list(EXAMPLES_POOL_OPTIONS.keys()),
                    value="data/train.json",
                    label="Examples Pool",
                    info="For Few-shot / Re-reading (FewShot)"
                )
            
            with gr.Row():
                plus_checkbox = gr.Checkbox(label="PS+ Mode", value=True, info="For Plan-and-Solve")
                re_add_method_dropdown = gr.Dropdown(
                    choices=["none", "0_CoT", "FewShot", "FewShot_CoT", "PaS"],
                    value="none",
                    label="Re-reading Add Method"
                )

    # --- GENERATION PARAMETERS ---
    with gr.Row():
        gr.Markdown("### 🎛️ Generation Parameters")
    with gr.Row():
        temp_slider = gr.Slider(0.0, 2.0, value=0.1, label="Temperature")
        top_p_slider = gr.Slider(0.0, 1.0, value=0.9, label="Top-p")
        max_tok_slider = gr.Slider(128, 2048, value=1024, label="Max Tokens", step=128)

    # --- INFERENCE ---
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Input")
            input_text = gr.Textbox(
                label="Vietnamese Text", 
                lines=6, 
                placeholder="Nhập văn bản tiếng Việt...",
                value="Tôi rất thích sản phẩm này nhưng nhân viên thái độ tệ quá."
            )
            run_btn = gr.Button("🚀 Run Inference", variant="primary", size="lg")
            
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            with gr.Tabs():
                with gr.TabItem("🕸️ SSA Graph"):
                    html_output = gr.HTML()
                
                with gr.TabItem("📄 JSON"):
                    json_output = gr.JSON()
                    
                with gr.TabItem("ℹ️ Metadata"):
                    meta_output = gr.Textbox(lines=18)

    # --- HISTORY ---
    with gr.Row():
        gr.Markdown("### 🕰️ History")
    with gr.Row():
        history_table = gr.Dataframe(
            headers=["ID", "Time", "Prompt", "Time", "Input"],
            datatype=["number", "str", "str", "str", "str"],
            wrap=True
        )

    # === EVENTS ===
    load_model_btn.click(
        fn=demo_mgr.load_model,
        inputs=[model_dropdown],
        outputs=[model_status]
    )
    
    run_btn.click(
        fn=demo_mgr.predict,
        inputs=[
            input_text, prompt_dropdown, n_shot_slider, examples_pool_dropdown,
            plus_checkbox, re_add_method_dropdown, temp_slider, top_p_slider, max_tok_slider
        ],
        outputs=[html_output, json_output, meta_output, history_table]
    )

    gr.Markdown("""
---
**Note:** 
- Few-shot & Re-reading (FewShot) require examples pool file
- Other techniques use hardcoded examples or no examples
- Adjust n_shot based on technique requirements
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)