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
              
    def load_model(self, model_name, dtype, device_map, trust_remote_code, 
                   init_extra_json, max_tokens, do_sample, temperature, 
                   top_p, top_k, gen_extra_json):
        """Load model with full config control."""
        if self.current_model_name == model_name and self.model is not None:
            return f"✅ Model '{model_name}' already loaded."
        
        try:
            # Parse extra args
            init_extras = json.loads(init_extra_json) if init_extra_json.strip() else {}
            gen_extras = json.loads(gen_extra_json) if gen_extra_json.strip() else {}
            
            # Build config
            config = {
                "model_id": model_name,
                "init_args": {
                    "model_id": model_name,
                    "dtype": dtype,
                    "device_map": device_map,
                    "trust_remote_code": trust_remote_code,
                    **init_extras
                },
                "generation_args": {
                    "max_new_tokens": int(max_tokens),
                    "do_sample": do_sample,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    **gen_extras
                }
            }
            
            self.model = HFModel(config)
            self.model.load_model()
            self.current_model_name = model_name
            return f"✅ Model '{model_name}' loaded!"
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def load_prompt_template(self, technique, n_shot, examples_pool, 
                            plus_mode, re_add_method):
        """Load prompt template with technique-specific params."""
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

    def predict(self, text, technique, n_shot, examples_pool, plus_mode, re_add_method):
        """Run inference."""
        if not self.model:
            return None, None, "⚠️ Load model first!", self.format_history()
        
        load_msg = self.load_prompt_template(technique, n_shot, examples_pool, plus_mode, re_add_method)
        if "❌" in load_msg or "⚠️" in load_msg:
            return None, None, load_msg, self.format_history()
            
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
            
            self.history.insert(0, {
                "id": len(self.history)+1,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": config_str,
                "exec": f"{exec_time:.2f}s",
                "text": text
            })
            
            metadata = f"""⏱️ Time: {exec_time:.2f}s (Gen: {gen_time:.2f}s)
🎯 Technique: {config_str}
🌱 Seed: {self.seed}

📝 System Prompt:
{system_prompt[:250]}...

💬 User Prompt:
{user_prompt[:250]}...

🤖 Response:
{raw_response[:350]}..."""
            
            return html_graph, result, metadata, self.format_history()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None, f"❌ Error: {str(e)}", self.format_history()

    def format_history(self):
        if not self.history:
            return []
        return [[r["id"], r["time"], r["config"], r["exec"], r["text"][:35]+"..."] for r in self.history]

# Initialize Manager
demo_mgr = DemoManager(seed=42)

# === GRADIO UI ===
with gr.Blocks(title="SSA Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Structured Sentiment Analysis Demo")
    
    # SEED
    with gr.Row(variant="compact"):
        seed_input = gr.Number(value=42, label="🌱 Random Seed", precision=0)
        seed_btn = gr.Button("Set", size="sm")
        seed_status = gr.Markdown("_Seed: 42_")
    
    # MODEL SETUP
    with gr.Accordion("⚙️ Model Configuration", open=True):
        with gr.Row():
            model_input = gr.Dropdown(
                choices=DEFAULT_MODEL_SUGGESTIONS,
                value=DEFAULT_MODEL_SUGGESTIONS[0],
                label="Model Name",
                allow_custom_value=True,
                info="Select or type custom model ID"
            )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Init Args**")
                dtype_dropdown = gr.Dropdown(["auto", "float16", "bfloat16", "float32"], value="auto", label="dtype")
                device_map_input = gr.Textbox(value="auto", label="device_map")
                trust_remote_code_check = gr.Checkbox(value=True, label="trust_remote_code")
                init_extra_json = gr.TextArea(
                    label="Extra Init Args (JSON)",
                    placeholder='{"attn_implementation": "flash_attention_2"}',
                    lines=2
                )
            
            with gr.Column():
                gr.Markdown("**Generation Args**")
                max_tokens_slider = gr.Slider(128, 4096, value=1024, step=128, label="max_new_tokens")
                do_sample_check = gr.Checkbox(value=False, label="do_sample")
                temperature_slider = gr.Slider(0.0, 2.0, value=0.1, label="temperature")
                top_p_slider = gr.Slider(0.0, 1.0, value=0.9, label="top_p")
                top_k_slider = gr.Slider(1, 100, value=50, step=1, label="top_k")
                gen_extra_json = gr.TextArea(
                    label="Extra Gen Args (JSON)",
                    placeholder='{"repetition_penalty": 1.1}',
                    lines=2
                )
        
        load_model_btn = gr.Button("🔄 Load Model", variant="primary", size="lg")
        model_status = gr.Markdown("_Ready_")
    
    # PROMPT CONFIG
    with gr.Accordion("🎯 Prompt Technique", open=True):
        technique_dropdown = gr.Dropdown(
            choices=list(PROMPT_TECHNIQUES.keys()),
            value="few_shot",
            label="Technique"
        )
        
        # Common params
        n_shot_slider = gr.Slider(0, 5, value=3, step=1, label="N-shot")
        examples_pool_dropdown = gr.Dropdown(choices=EXAMPLES_POOLS, value=EXAMPLES_POOLS[0], label="Examples Pool")
        
        # Technique-specific params
        plus_mode_check = gr.Checkbox(value=True, label="PS+ Mode (Plan-and-Solve)")
        re_add_method_dropdown = gr.Dropdown(
            choices=["none", "0_CoT", "FewShot", "FewShot_CoT", "PaS"],
            value="none",
            label="Add Method (Re-reading)"
        )
    
    # INFERENCE
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Input")
            input_text = gr.Textbox(
                label="Text",
                lines=6,
                value="Tôi rất thích sản phẩm này nhưng nhân viên thái độ tệ quá."
            )
            run_btn = gr.Button("🚀 Run", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            with gr.Tabs():
                with gr.TabItem("🕸️ Graph"):
                    html_output = gr.HTML()
                with gr.TabItem("📄 JSON"):
                    json_output = gr.JSON()
                with gr.TabItem("ℹ️ Meta"):
                    meta_output = gr.Textbox(lines=15)
    
    # HISTORY
    gr.Markdown("### 🕰️ History")
    history_table = gr.Dataframe(
        headers=["ID", "Time", "Technique", "Exec", "Input"],
        datatype=["number", "str", "str", "str", "str"]
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
    
    # === EVENTS ===
    seed_btn.click(fn=demo_mgr.setup_seed, inputs=[seed_input], outputs=[seed_status])
    
    load_model_btn.click(
        fn=demo_mgr.load_model,
        inputs=[
            model_input, dtype_dropdown, device_map_input, trust_remote_code_check,
            init_extra_json, max_tokens_slider, do_sample_check, temperature_slider,
            top_p_slider, top_k_slider, gen_extra_json
        ],
        outputs=[model_status]
    )
    
    run_btn.click(
        fn=demo_mgr.predict,
        inputs=[input_text, technique_dropdown, n_shot_slider, examples_pool_dropdown, plus_mode_check, re_add_method_dropdown],
        outputs=[html_output, json_output, meta_output, history_table]
    )
    
    gr.Markdown("""
---
**Tips:**
- 💡 Type any HuggingFace model ID in Model Name
- 🔧 Use Extra Args JSON for advanced params
- 🎯 Params auto-hide based on technique
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)