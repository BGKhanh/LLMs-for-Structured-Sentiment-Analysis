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
DEFAULT_MODEL_SUGGESTIONS = [
    "google/gemma-3-4b-it",
    "Qwen/Qwen3-4B-Instruct-2507",
    "meta-llama/Llama-3.2-3B-Instruct",
]

PROMPT_TECHNIQUES = {
    "few_shot": {"name": "Few-shot", "params": ["n_shot", "examples_pool"]},
    "few_shot_cot": {"name": "Few-shot CoT", "params": ["n_shot"]},
    "plan_and_solve": {"name": "Plan-and-Solve", "params": ["plus_mode", "n_shot_conditional"]},
    "re_reading": {"name": "Re-reading", "params": ["add_method", "n_shot_conditional", "examples_pool_conditional"]},
    "re2_pas_cot": {"name": "RE2+PaS+CoT", "params": ["n_shot"]}
}

EXAMPLES_POOLS = ["data/train.json", "data/dev.json", "data/test.json"]


# === DEMO MANAGER CLASS ===
class DemoManager:
    def __init__(self, seed=42):
        self.seed = seed
        self.setup_seed(seed)
        
        self.model = None
        self.current_model_name = None
        self.model_config = None  # Store init config separately
        self.prompt_template = None
        self.current_prompt_config = None
        self.history = []
    
    def setup_seed(self, seed):
        try:
            setup_reproducible_environment(seed=seed)
            self.seed = seed
            return f"✅ Seed set to {seed}"
        except Exception as e:
            return f"⚠️ Warning: {str(e)}"
              
    def load_model(self, model_name, dtype, device_map, trust_remote_code, init_extra_json):
        """Load model ONCE with init params only."""
        if self.current_model_name == model_name and self.model is not None:
            return f"✅ Model '{model_name}' already loaded. Use 'Update Gen Params' to change generation settings."
        
        try:
            # Parse extra init args
            init_extras = json.loads(init_extra_json) if init_extra_json.strip() else {}
            
            # Build INIT-ONLY config
            self.model_config = {
                "model_id": model_name,
                "init_args": {
                    "model_id": model_name,
                    "dtype": dtype,
                    "device_map": device_map,
                    "trust_remote_code": trust_remote_code,
                    **init_extras
                },
                "generation_args": {
                    # Default generation args (will be updated later)
                    "max_new_tokens": 1024,
                    "do_sample": False,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 50
                }
            }
            
            self.model = HFModel(self.model_config)
            self.model.load_model()
            self.current_model_name = model_name
            return f"✅ Model '{model_name}' loaded successfully!\n💡 Now you can update generation parameters anytime."
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error loading model: {str(e)}"
    
    def update_generation_params(self, max_tokens, do_sample, temperature, top_p, top_k, gen_extra_json):
        """Update generation params WITHOUT reloading model."""
        if not self.model:
            return "⚠️ Load model first!"
        
        try:
            gen_extras = json.loads(gen_extra_json) if gen_extra_json.strip() else {}
            
            # Update generation args in model config
            self.model_config["generation_args"] = {
                "max_new_tokens": int(max_tokens),
                "do_sample": do_sample,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                **gen_extras
            }
            
            # Update model's generation config
            self.model.config = self.model_config
            
            return f"✅ Generation parameters updated!\n📊 Params: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}, top_k={top_k}, do_sample={do_sample}"
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def load_prompt_template(self, technique, n_shot, examples_pool, plus_mode, re_add_method):
        """Load prompt template with technique-specific params."""
        config_key = f"{technique}_{n_shot}_{examples_pool}_{plus_mode}_{re_add_method}"
        
        if self.current_prompt_config == config_key and self.prompt_template:
            return "✅ Prompt template ready"
        
        try:
            if technique == "few_shot":
                if n_shot > 0 and not Path(examples_pool).exists():
                    return f"⚠️ Examples pool not found: {examples_pool}"
                self.prompt_template = FewShotPrompt(
                    eng=False, n_shot=n_shot,
                    examples_pool_path=examples_pool if n_shot > 0 else None
                )
                
            elif technique == "few_shot_cot":
                self.prompt_template = FewShotCoTPrompt(eng=False, n_shot=n_shot)
                
            elif technique == "plan_and_solve":
                self.prompt_template = PlanAndSolvePrompt(
                    eng=False, plus=plus_mode,
                    n_shot=n_shot if plus_mode else 0
                )
                
            elif technique == "re_reading":
                needs_pool = re_add_method == "FewShot"
                if needs_pool and n_shot > 0 and not Path(examples_pool).exists():
                    return f"⚠️ Examples pool not found: {examples_pool}"
                self.prompt_template = ReReadingPrompt(
                    eng=False, add_method=re_add_method, n_shot=n_shot,
                    examples_pool_path=examples_pool if needs_pool and n_shot > 0 else None
                )
                
            elif technique == "re2_pas_cot":
                self.prompt_template = Re2PaSCoTPrompt(eng=False, n_shot=n_shot)
            
            self.prompt_template.prepare()
            self.current_prompt_config = config_key
            return f"✅ {PROMPT_TECHNIQUES[technique]['name']} template loaded"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error loading prompt: {str(e)}"

    def predict(self, text, technique, n_shot, examples_pool, plus_mode, re_add_method):
        """Run inference with current settings."""
        if not self.model:
            return None, None, None, None, "⚠️ Load model first!", self.format_history()
        
        # Load/verify prompt template
        load_msg = self.load_prompt_template(technique, n_shot, examples_pool, plus_mode, re_add_method)
        if "❌" in load_msg or "⚠️" in load_msg:
            return None, None, None, None, load_msg, self.format_history()
            
        try:
            # Generate
            start = time.time()
            system_prompt, user_prompt = self.prompt_template.get_prompt(text, f"demo_{len(self.history)+1}")
            raw_response, gen_time = self.model.generate_single(system_prompt, user_prompt)
            json_str = self.model.extract_response(raw_response)
            processed = postprocess_response(json_str, text, f"demo_{len(self.history)+1}")
            result = json.loads(processed)
            exec_time = time.time() - start
            
            # Create visualization
            html_graph = convert_ssa_to_spacy(text, result)
            
            # Format config string
            config_str = PROMPT_TECHNIQUES[technique]['name']
            if n_shot > 0:
                config_str += f" ({n_shot}-shot)"
            
            # Count tokens (basic approximation)
            input_tokens = len(system_prompt.split()) + len(user_prompt.split())
            output_tokens = len(raw_response.split())
            
            # Build metadata
            gen_params = self.model_config["generation_args"]
            metadata = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "execution_time": f"{exec_time:.3f}s",
                "generation_time": f"{gen_time:.3f}s",
                "seed": self.seed,
                "model": self.current_model_name,
                "technique": config_str,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "generation_params": gen_params
            }
            
            # Format metadata display
            metadata_display = f"""⏱️ **Execution Time:** {exec_time:.3f}s (Generation: {gen_time:.3f}s)
📊 **Tokens:** Input={input_tokens} | Output={output_tokens}
🎯 **Technique:** {config_str}
🤖 **Model:** {self.current_model_name}
🌱 **Seed:** {self.seed}

⚙️ **Generation Parameters:**
• max_new_tokens: {gen_params['max_new_tokens']}
• temperature: {gen_params['temperature']}
• top_p: {gen_params['top_p']}
• top_k: {gen_params['top_k']}
• do_sample: {gen_params['do_sample']}
"""
            
            # Add to history
            self.history.insert(0, {
                "id": len(self.history) + 1,
                "time": metadata["timestamp"],
                "config": config_str,
                "exec": f"{exec_time:.2f}s",
                "text": text,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": raw_response,
                "result": result,
                "metadata": metadata
            })
            
            # Return: graph, json, full_input, raw_response, metadata, history
            full_input = f"**System Prompt:**\n{system_prompt}\n\n**User Prompt:**\n{user_prompt}"
            
            return html_graph, result, full_input, raw_response, metadata_display, self.format_history()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None, None, None, f"❌ Inference error: {str(e)}", self.format_history()

    def format_history(self):
        """Format history for display."""
        if not self.history:
            return []
        return [[r["id"], r["time"], r["config"], r["exec"], r["text"][:40]+"..."] for r in self.history]

# Initialize Manager
demo_mgr = DemoManager(seed=42)

# === GRADIO UI ===
with gr.Blocks(title="SSA Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Structured Sentiment Analysis Demo")
    
    # === SEED ===
    with gr.Row(variant="compact"):
        seed_input = gr.Number(value=42, label="🌱 Random Seed", precision=0)
        seed_btn = gr.Button("Set Seed", size="sm")
        seed_status = gr.Markdown("_Seed: 42_")
    
    # === MODEL SETUP (Load Once) ===
    with gr.Accordion("🔧 Model Setup (Load Once)", open=True):
        gr.Markdown("**Initialize model with these parameters. Model loads ONCE per session.**")
        
        model_input = gr.Dropdown(
            choices=DEFAULT_MODEL_SUGGESTIONS,
            value=DEFAULT_MODEL_SUGGESTIONS[0],
            label="Model Name",
            allow_custom_value=True,
            info="Select or type custom HuggingFace model ID"
        )
        
        with gr.Row():
            dtype_dropdown = gr.Dropdown(
                ["auto", "float16", "bfloat16", "float32"], 
                value="auto", 
                label="dtype"
            )
            device_map_input = gr.Textbox(value="auto", label="device_map")
            trust_remote_code_check = gr.Checkbox(value=True, label="trust_remote_code")
        
        init_extra_json = gr.TextArea(
            label="Extra Init Args (JSON)",
            placeholder='{"attn_implementation": "flash_attention_2"}',
            lines=2
        )
        
        load_model_btn = gr.Button("🔄 Load Model", variant="primary", size="lg")
        model_status = gr.Markdown("_No model loaded_")
    
    # === GENERATION PARAMS (Update Anytime) ===
    with gr.Accordion("⚙️ Generation Parameters (Update Anytime)", open=True):
        gr.Markdown("**Change these parameters anytime without reloading the model.**")
        
        with gr.Row():
            max_tokens_slider = gr.Slider(128, 4096, value=1024, step=128, label="max_new_tokens")
            temperature_slider = gr.Slider(0.0, 2.0, value=0.1, step=0.05, label="temperature")
            top_p_slider = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="top_p")
        
        with gr.Row():
            top_k_slider = gr.Slider(1, 100, value=50, step=1, label="top_k")
            do_sample_check = gr.Checkbox(value=False, label="do_sample")
        
        gen_extra_json = gr.TextArea(
            label="Extra Generation Args (JSON)",
            placeholder='{"repetition_penalty": 1.1}',
            lines=2
        )
        
        update_gen_btn = gr.Button("🔄 Update Generation Params", variant="secondary")
        gen_status = gr.Markdown("_Default params active_")
    
    # === PROMPT CONFIG ===
    with gr.Accordion("🎯 Prompt Technique", open=True):
        technique_dropdown = gr.Dropdown(
            choices=list(PROMPT_TECHNIQUES.keys()),
            value="few_shot",
            label="Technique"
        )
        
        n_shot_slider = gr.Slider(0, 5, value=3, step=1, label="N-shot")
        examples_pool_dropdown = gr.Dropdown(
            choices=EXAMPLES_POOLS, 
            value=EXAMPLES_POOLS[0], 
            label="Examples Pool"
        )
        
        plus_mode_check = gr.Checkbox(value=True, label="PS+ Mode (Plan-and-Solve)")
        re_add_method_dropdown = gr.Dropdown(
            choices=["none", "0_CoT", "FewShot", "FewShot_CoT", "PaS"],
            value="none",
            label="Add Method (Re-reading)"
        )
    
    # === INFERENCE ===
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Input")
            input_text = gr.Textbox(
                label="Text",
                lines=8,
                value="Tôi rất thích sản phẩm này nhưng nhân viên thái độ tệ quá."
            )
            run_btn = gr.Button("🚀 Run Inference", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            with gr.Tabs():
                with gr.TabItem("🕸️ Visualization"):
                    html_output = gr.HTML()
                
                with gr.TabItem("📄 JSON Output"):
                    json_output = gr.JSON()
                
                with gr.TabItem("📥 Full Input"):
                    full_input_output = gr.Textbox(lines=15, show_copy_button=True)
                
                with gr.TabItem("🤖 Raw Response"):
                    raw_response_output = gr.Textbox(lines=15, show_copy_button=True)
                
                with gr.TabItem("ℹ️ Metadata"):
                    metadata_output = gr.Markdown()
    
    # === HISTORY ===
    gr.Markdown("### 🕰️ Run History")
    history_table = gr.Dataframe(
        headers=["ID", "Timestamp", "Technique", "Time", "Input Preview"],
        datatype=["number", "str", "str", "str", "str"],
        interactive=False
    )
    
    # === DYNAMIC UI ===
    def update_prompt_params(technique):
        """Show/hide params based on technique."""
        params = PROMPT_TECHNIQUES[technique]["params"]
        return {
            n_shot_slider: gr.update(visible="n_shot" in params or "n_shot_conditional" in params),
            examples_pool_dropdown: gr.update(visible="examples_pool" in params or "examples_pool_conditional" in params),
            plus_mode_check: gr.update(visible="plus_mode" in params),
            re_add_method_dropdown: gr.update(visible="add_method" in params)
        }
    
    technique_dropdown.change(
        fn=update_prompt_params,
        inputs=[technique_dropdown],
        outputs=[n_shot_slider, examples_pool_dropdown, plus_mode_check, re_add_method_dropdown]
    )
    
    # === EVENT HANDLERS ===
    seed_btn.click(
        fn=demo_mgr.setup_seed, 
        inputs=[seed_input], 
        outputs=[seed_status]
    )
    
    load_model_btn.click(
        fn=demo_mgr.load_model,
        inputs=[
            model_input, dtype_dropdown, device_map_input, 
            trust_remote_code_check, init_extra_json
        ],
        outputs=[model_status]
    )
    
    update_gen_btn.click(
        fn=demo_mgr.update_generation_params,
        inputs=[
            max_tokens_slider, do_sample_check, temperature_slider,
            top_p_slider, top_k_slider, gen_extra_json
        ],
        outputs=[gen_status]
    )
    
    run_btn.click(
        fn=demo_mgr.predict,
        inputs=[
            input_text, technique_dropdown, n_shot_slider, 
            examples_pool_dropdown, plus_mode_check, re_add_method_dropdown
        ],
        outputs=[
            html_output, json_output, full_input_output, 
            raw_response_output, metadata_output, history_table
        ]
    )
    
    gr.Markdown("""
---
**💡 Tips:**
- Load model ONCE at the start of your session
- Change generation parameters anytime without reloading
- Each technique has different parameter requirements
- All runs are saved in history for comparison
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)