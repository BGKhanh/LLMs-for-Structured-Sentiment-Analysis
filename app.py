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

# === PARAMETER REGISTRY ===
from transformers import GenerationConfig, AutoConfig

# Build generation param registry
GENERATION_PARAM_REGISTRY = {}
try:
    gen_config = GenerationConfig()
    for name, value in gen_config.to_dict().items():
        if name not in ['transformers_version', '_from_model_config']:
            GENERATION_PARAM_REGISTRY[name] = {
                "default": value,
                "type": type(value).__name__
            }
except Exception as e:
    print(f"Warning: Could not build generation registry: {e}")

# Common init params (manually curated for safety)
INIT_PARAM_REGISTRY = {
    "attn_implementation": {"default": None, "type": "str", "choices": [None, "flash_attention_2", "sdpa", "eager"]},
    "torch_dtype": {"default": None, "type": "str", "choices": [None, "float16", "bfloat16", "float32"]},
    "low_cpu_mem_usage": {"default": True, "type": "bool"},
    "use_cache": {"default": True, "type": "bool"},
    "rope_scaling": {"default": None, "type": "dict"},
}

# === GLOBAL CONFIG ===
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
        setup_reproducible_environment(seed=seed)
        
        self.model = None
        self.current_model_name = None
        self.model_config = None
        self.prompt_template = None
        self.current_prompt_config = None
        self.history = []
        
        # Extra args state
        self.active_init_extra_args = {}
        self.active_gen_extra_args = {}
    
    # === EXTRA ARGS MANAGEMENT ===
    def add_init_param(self, param_name):
        """Add init param to active list."""
        if not param_name or param_name in self.active_init_extra_args:
            return self.render_init_extra_args()
        
        if param_name in INIT_PARAM_REGISTRY:
            self.active_init_extra_args[param_name] = INIT_PARAM_REGISTRY[param_name]["default"]
        
        return self.render_init_extra_args()
    
    def remove_init_param(self, param_name):
        """Remove init param from active list."""
        self.active_init_extra_args.pop(param_name, None)
        return self.render_init_extra_args()
    
    def update_init_param_value(self, param_name, value):
        """Update init param value."""
        if param_name in self.active_init_extra_args:
            self.active_init_extra_args[param_name] = value
        return self.render_init_extra_args()
    
    def add_gen_param(self, param_name):
        """Add generation param to active list."""
        if not param_name or param_name in self.active_gen_extra_args:
            return self.render_gen_extra_args()
        
        if param_name in GENERATION_PARAM_REGISTRY:
            self.active_gen_extra_args[param_name] = GENERATION_PARAM_REGISTRY[param_name]["default"]
        
        return self.render_gen_extra_args()
    
    def remove_gen_param(self, param_name):
        """Remove generation param from active list."""
        self.active_gen_extra_args.pop(param_name, None)
        return self.render_gen_extra_args()
    
    def update_gen_param_value(self, param_name, value):
        """Update generation param value."""
        if param_name in self.active_gen_extra_args:
            self.active_gen_extra_args[param_name] = value
        return self.render_gen_extra_args()
    
    def render_init_extra_args(self):
        """Render active init extra args as markdown."""
        if not self.active_init_extra_args:
            return "_No extra init args added_"
        
        lines = ["**Active Extra Init Args:**\n"]
        for name, value in self.active_init_extra_args.items():
            lines.append(f"• `{name}`: {value}")
        return "\n".join(lines)
    
    def render_gen_extra_args(self):
        """Render active gen extra args as markdown."""
        if not self.active_gen_extra_args:
            return "_No extra generation args added_"
        
        lines = ["**Active Extra Generation Args:**\n"]
        for name, value in self.active_gen_extra_args.items():
            lines.append(f"• `{name}`: {value}")
        return "\n".join(lines)
    
    # === MODEL MANAGEMENT ===
    def load_model(self, model_name, dtype, device_map, trust_remote_code):
        """Load model ONCE with init params."""
        if self.current_model_name == model_name and self.model is not None:
            return f"✅ Model '{model_name}' already loaded."
        
        try:
            # Build init args with extra params
            init_args = {
                "model_id": model_name,
                "dtype": dtype,
                "device_map": device_map,
                "trust_remote_code": trust_remote_code,
                **self.active_init_extra_args
            }
            
            self.model_config = {
                "model_id": model_name,
                "init_args": init_args,
                "generation_args": {
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
            
            extra_info = f"\n🔧 Extra init args: {list(self.active_init_extra_args.keys())}" if self.active_init_extra_args else ""
            return f"✅ Model '{model_name}' loaded!{extra_info}"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}"
    
    def update_generation_params(self, max_tokens, do_sample, temperature, top_p, top_k):
        """Update generation params WITHOUT reloading model."""
        if not self.model:
            return "⚠️ Load model first!"
        
        try:
            # Build generation args with extra params
            gen_args = {
                "max_new_tokens": int(max_tokens),
                "do_sample": do_sample,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                **self.active_gen_extra_args
            }
            
            self.model_config["generation_args"] = gen_args
            self.model.config = self.model_config
            
            extra_info = f"\n🔧 Extra gen args: {list(self.active_gen_extra_args.keys())}" if self.active_gen_extra_args else ""
            return f"✅ Generation params updated!{extra_info}"
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    # === PROMPT TEMPLATE ===
    def load_prompt_template(self, technique, n_shot, examples_pool, plus_mode, re_add_method):
        """Load prompt template."""
        config_key = f"{technique}_{n_shot}_{examples_pool}_{plus_mode}_{re_add_method}"
        
        if self.current_prompt_config == config_key and self.prompt_template:
            return "✅ Prompt ready"
        
        try:
            if technique == "few_shot":
                if n_shot > 0 and not Path(examples_pool).exists():
                    return f"⚠️ Pool not found: {examples_pool}"
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
                    return f"⚠️ Pool not found: {examples_pool}"
                self.prompt_template = ReReadingPrompt(
                    eng=False, add_method=re_add_method, n_shot=n_shot,
                    examples_pool_path=examples_pool if needs_pool and n_shot > 0 else None
                )
                
            elif technique == "re2_pas_cot":
                self.prompt_template = Re2PaSCoTPrompt(eng=False, n_shot=n_shot)
            
            self.prompt_template.prepare()
            self.current_prompt_config = config_key
            return f"✅ {PROMPT_TECHNIQUES[technique]['name']} ready"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}"

    # === INFERENCE ===
    def predict(self, text, technique, n_shot, examples_pool, plus_mode, re_add_method):
        """Run inference."""
        if not self.model:
            return None, None, None, None, "⚠️ Load model first!", self.format_history()
        
        load_msg = self.load_prompt_template(technique, n_shot, examples_pool, plus_mode, re_add_method)
        if "❌" in load_msg or "⚠️" in load_msg:
            return None, None, None, None, load_msg, self.format_history()
            
        try:
            start = time.time()
            system_prompt, user_prompt = self.prompt_template.get_prompt(text, f"demo_{len(self.history)+1}")
            raw_response, gen_time = self.model.generate_single(system_prompt, user_prompt)
            json_str = self.model.extract_response(raw_response)
            processed = postprocess_response(json_str, text, f"demo_{len(self.history)+1}")
            result = json.loads(processed)
            exec_time = time.time() - start
            
            html_graph = convert_ssa_to_spacy(text, result)
            
            config_str = PROMPT_TECHNIQUES[technique]['name']
            if n_shot > 0:
                config_str += f" ({n_shot}-shot)"
            
            input_tokens = len(system_prompt.split()) + len(user_prompt.split())
            output_tokens = len(raw_response.split())
            
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
            
            metadata_display = f"""⏱️ **Execution Time:** {exec_time:.3f}s (Gen: {gen_time:.3f}s)
📊 **Tokens:** Input={input_tokens} | Output={output_tokens}
🎯 **Technique:** {config_str}
🤖 **Model:** {self.current_model_name}
🌱 **Seed:** {self.seed}

⚙️ **Generation Parameters:**
"""
            for k, v in gen_params.items():
                metadata_display += f"• {k}: {v}\n"
            
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
            
            full_input = f"**System Prompt:**\n{system_prompt}\n\n**User Prompt:**\n{user_prompt}"
            
            return html_graph, result, full_input, raw_response, metadata_display, self.format_history()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None, None, None, f"❌ Error: {str(e)}", self.format_history()

    def format_history(self):
        if not self.history:
            return []
        return [[r["id"], r["time"], r["config"], r["exec"], r["text"][:40]+"..."] for r in self.history]

# Initialize Manager
demo_mgr = DemoManager(seed=42)

# === GRADIO UI ===
with gr.Blocks(title="SSA Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Structured Sentiment Analysis Demo")
    
    # === MODEL SETUP & GENERATION PARAMS (SIDE BY SIDE) ===
    with gr.Row(equal_height=True):
        # LEFT: Model Setup
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 Model Setup\n_Load once per session_")
            
            model_input = gr.Dropdown(
                choices=DEFAULT_MODEL_SUGGESTIONS,
                value=DEFAULT_MODEL_SUGGESTIONS[0],
                label="Model Name",
                allow_custom_value=True
            )
            
            dtype_dropdown = gr.Dropdown(
                ["auto", "float16", "bfloat16", "float32"], 
                value="auto", 
                label="dtype"
            )
            device_map_input = gr.Textbox(value="auto", label="device_map")
            trust_remote_code_check = gr.Checkbox(value=True, label="trust_remote_code")
            
            # Extra Init Args
            gr.Markdown("**Extra Init Args**")
            init_param_search = gr.Dropdown(
                choices=list(INIT_PARAM_REGISTRY.keys()),
                label="🔍 Search parameter",
                value=None
            )
            with gr.Row():
                add_init_btn = gr.Button("➕ Add", size="sm", scale=1)
                init_param_value = gr.Textbox(label="Value", scale=2, placeholder="Enter value...")
                remove_init_btn = gr.Button("❌ Remove", size="sm", scale=1)
            
            init_extra_display = gr.Markdown("_No extra init args_")
            
            load_model_btn = gr.Button("🔄 Load Model", variant="primary", size="lg")
            model_status = gr.Markdown("_No model loaded_")
        
        # RIGHT: Generation Params
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Generation Parameters\n_Update anytime_")
            
            max_tokens_slider = gr.Slider(128, 4096, value=1024, step=128, label="max_new_tokens")
            temperature_slider = gr.Slider(0.0, 2.0, value=0.1, step=0.05, label="temperature")
            top_p_slider = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="top_p")
            top_k_slider = gr.Slider(1, 100, value=50, step=1, label="top_k")
            do_sample_check = gr.Checkbox(value=False, label="do_sample")
            
            # Extra Gen Args
            gr.Markdown("**Extra Generation Args**")
            gen_param_search = gr.Dropdown(
                choices=list(GENERATION_PARAM_REGISTRY.keys()),
                label="🔍 Search parameter",
                value=None
            )
            with gr.Row():
                add_gen_btn = gr.Button("➕ Add", size="sm", scale=1)
                gen_param_value = gr.Textbox(label="Value", scale=2, placeholder="Enter value...")
                remove_gen_btn = gr.Button("❌ Remove", size="sm", scale=1)
            
            gen_extra_display = gr.Markdown("_No extra gen args_")
            
            update_gen_btn = gr.Button("🔄 Update Generation Params", variant="secondary", size="lg")
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
        
        plus_mode_check = gr.Checkbox(value=True, label="PS+ Mode")
        re_add_method_dropdown = gr.Dropdown(
            choices=["none", "0_CoT", "FewShot", "FewShot_CoT", "PaS"],
            value="none",
            label="Add Method"
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
                
                with gr.TabItem("📄 JSON"):
                    json_output = gr.JSON()
                
                with gr.TabItem("📥 Full Input"):
                    full_input_output = gr.Textbox(lines=15, max_lines=20)
                
                with gr.TabItem("🤖 Raw Response"):
                    raw_response_output = gr.Textbox(lines=15, max_lines=20)
                
                with gr.TabItem("ℹ️ Metadata"):
                    metadata_output = gr.Markdown()
    
    # === HISTORY ===
    gr.Markdown("### 🕰️ Run History")
    history_table = gr.Dataframe(
        headers=["ID", "Time", "Technique", "Exec", "Input"],
        datatype=["number", "str", "str", "str", "str"]
    )
    
    # === DYNAMIC UI ===
    def update_prompt_params(technique):
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
    
    # Init extra args
    add_init_btn.click(
        fn=lambda name, val: demo_mgr.update_init_param_value(name, val) if name and val else demo_mgr.add_init_param(name),
        inputs=[init_param_search, init_param_value],
        outputs=[init_extra_display]
    )
    
    remove_init_btn.click(
        fn=demo_mgr.remove_init_param,
        inputs=[init_param_search],
        outputs=[init_extra_display]
    )
    
    # Gen extra args
    add_gen_btn.click(
        fn=lambda name, val: demo_mgr.update_gen_param_value(name, val) if name and val else demo_mgr.add_gen_param(name),
        inputs=[gen_param_search, gen_param_value],
        outputs=[gen_extra_display]
    )
    
    remove_gen_btn.click(
        fn=demo_mgr.remove_gen_param,
        inputs=[gen_param_search],
        outputs=[gen_extra_display]
    )
    
    # Model & generation
    load_model_btn.click(
        fn=demo_mgr.load_model,
        inputs=[model_input, dtype_dropdown, device_map_input, trust_remote_code_check],
        outputs=[model_status]
    )
    
    update_gen_btn.click(
        fn=demo_mgr.update_generation_params,
        inputs=[max_tokens_slider, do_sample_check, temperature_slider, top_p_slider, top_k_slider],
        outputs=[gen_status]
    )
    
    # Inference
    run_btn.click(
        fn=demo_mgr.predict,
        inputs=[input_text, technique_dropdown, n_shot_slider, examples_pool_dropdown, plus_mode_check, re_add_method_dropdown],
        outputs=[html_output, json_output, full_input_output, raw_response_output, metadata_output, history_table]
    )
    
    gr.Markdown("""
---
**💡 Tips:**
- Load model once, update generation params anytime
- Use extra args to add advanced parameters
- Search parameters from dropdown, add with value, remove when not needed
- All runs saved in history
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)