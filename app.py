# app.py

import gradio as gr
import json
import time
import os
import sys
import inspect
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
from transformers import GenerationConfig

# FROM_PRETRAINED PARAMS - Extracted from HuggingFace docs
# Source: https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
INFRA_LOAD_PARAMS = {
    # Basic loading
    "pretrained_model_name_or_path": {"default": None, "type": "str", "group": "Basic", "required": True},
    "config": {"default": None, "type": "Union[PretrainedConfig, str]", "group": "Basic"},
    "cache_dir": {"default": None, "type": "str", "group": "Cache"},
    "force_download": {"default": False, "type": "bool", "group": "Cache"},
    "local_files_only": {"default": False, "type": "bool", "group": "Cache"},
    "token": {"default": None, "type": "Union[str, bool]", "group": "Auth"},
    "revision": {"default": "main", "type": "str", "group": "Version"},
    "use_safetensors": {"default": None, "type": "bool", "group": "Loading"},
    "weights_only": {"default": True, "type": "bool", "group": "Loading"},
    
    # TF/Flax loading
    "from_tf": {"default": False, "type": "bool", "group": "Framework"},
    "from_flax": {"default": False, "type": "bool", "group": "Framework"},
    
    # Model structure
    "state_dict": {"default": None, "type": "dict", "group": "Advanced"},
    "ignore_mismatched_sizes": {"default": False, "type": "bool", "group": "Advanced"},
    
    # Network & proxies
    "proxies": {"default": None, "type": "dict", "group": "Network"},
    
    # Output control
    "output_loading_info": {"default": False, "type": "bool", "group": "Debug"},
    
    # Attention implementation
    "attn_implementation": {
        "default": None, 
        "type": "str", 
        "group": "Performance",
        "choices": ["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
        "description": "Attention implementation: eager (manual), sdpa (PyTorch), flash_attention_2/3"
    },
    
    # Device & dtype - BIG MODEL INFERENCE
    "dtype": {
        "default": None, 
        "type": "str", 
        "group": "Device",
        "choices": ["auto", "float16", "bfloat16", "float32"],
        "description": "Override torch_dtype. 'auto' uses config, else loads in specified dtype"
    },
    "device_map": {
        "default": None, 
        "type": "Union[str, dict]", 
        "group": "Device",
        "description": "Device mapping: 'auto', 'cpu', 'cuda:0', or dict mapping modules to devices"
    },
    "max_memory": {
        "default": None, 
        "type": "dict", 
        "group": "Device",
        "description": "Dict of device:max_memory (e.g., {0: '10GB', 'cpu': '30GB'})"
    },
    "tp_plan": {
        "default": None, 
        "type": "str", 
        "group": "Distributed",
        "description": "Tensor parallel plan, currently only 'auto'"
    },
    "tp_size": {
        "default": None, 
        "type": "int", 
        "group": "Distributed",
        "description": "Tensor parallel degree"
    },
    "device_mesh": {
        "default": None, 
        "type": "torch.distributed.DeviceMesh", 
        "group": "Distributed",
        "description": "Device mesh for tensor parallelism"
    },
    "offload_folder": {
        "default": None, 
        "type": "str", 
        "group": "Memory",
        "description": "Folder for offloading weights to disk if device_map contains 'disk'"
    },
    "offload_buffers": {
        "default": False, 
        "type": "bool", 
        "group": "Memory",
        "description": "Whether to offload buffers with model parameters"
    },
    
    # Quantization
    "quantization_config": {
        "default": None, 
        "type": "Union[QuantizationConfigMixin, dict]", 
        "group": "Quantization",
        "description": "Quantization config (bitsandbytes, GPTQ, etc.)"
    },
    "load_in_4bit": {
        "default": False, 
        "type": "bool", 
        "group": "Quantization",
        "description": "Load model in 4-bit (bitsandbytes)"
    },
    "load_in_8bit": {
        "default": False, 
        "type": "bool", 
        "group": "Quantization",
        "description": "Load model in 8-bit (bitsandbytes)"
    },
    
    # File structure
    "subfolder": {
        "default": "", 
        "type": "str", 
        "group": "Loading",
        "description": "Subfolder in model repo where files are located"
    },
    "variant": {
        "default": None, 
        "type": "str", 
        "group": "Loading",
        "description": "Load weights from variant filename (e.g., 'fp16')"
    },
    
    # Safety & trust
    "trust_remote_code": {
        "default": False, 
        "type": "bool", 
        "group": "Safety",
        "description": "Allow loading custom code from model hub"
    },
    
    # Advanced loading
    "key_mapping": {
        "default": None, 
        "type": "dict", 
        "group": "Advanced",
        "description": "Mapping of weight names for compatible architectures"
    },
    
    # Low-level device control
    "torch_dtype": {
        "default": None, 
        "type": "str", 
        "group": "Device",
        "choices": ["float16", "bfloat16", "float32"],
        "description": "Torch dtype (prefer using 'dtype' parameter)"
    },
    "low_cpu_mem_usage": {
        "default": False, 
        "type": "bool", 
        "group": "Memory",
        "description": "Load model in low CPU memory mode"
    },
}

# RUNTIME/FORWARD PARAMS - Safe params that can be set after loading
RUNTIME_FORWARD_PARAMS = {
    "use_cache": {"default": True, "type": "bool", "description": "Use KV cache for generation"},
    "output_hidden_states": {"default": False, "type": "bool", "description": "Return all hidden states"},
    "output_attentions": {"default": False, "type": "bool", "description": "Return attention weights"},
    "return_dict": {"default": True, "type": "bool", "description": "Return ModelOutput instead of tuple"},
}

# Combine for extra args dropdown
INIT_PARAM_REGISTRY = {**INFRA_LOAD_PARAMS, **RUNTIME_FORWARD_PARAMS}

# Auto-discover generation params from GenerationConfig
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
    def add_init_param(self, param_name, value_str):
        """Add/update init param."""
        if not param_name:
            return self.get_init_extra_list(), self.render_init_extra_args()
        
        # Get param info from registry
        param_info = INIT_PARAM_REGISTRY.get(param_name, {})
        param_type = param_info.get("type", "str")
        
        # Parse value based on type
        try:
            if param_type == "bool":
                # Handle bool: true/false/True/False/1/0
                value = value_str.lower() in ['true', '1', 'yes']
            elif param_type == "int":
                value = int(value_str)
            elif param_type == "float":
                value = float(value_str)
            elif param_type == "str":
                if value_str.lower() == "none":
                    value = None
                else:
                    value = value_str
            else:
                # Try JSON parse as fallback
                value = json.loads(value_str)
        except:
            # Fallback to string
            value = value_str if value_str else None
        
        self.active_init_extra_args[param_name] = value
        return self.get_init_extra_list(), self.render_init_extra_args()
    
    def remove_init_param(self, param_name):
        """Remove init param."""
        if param_name in self.active_init_extra_args:
            self.active_init_extra_args.pop(param_name)
        return self.get_init_extra_list(), self.render_init_extra_args()
    
    def add_gen_param(self, param_name, value_str):
        """Add/update generation param."""
        if not param_name:
            return self.get_gen_extra_list(), self.render_gen_extra_args()
        
        # Get param info from registry
        param_info = GENERATION_PARAM_REGISTRY.get(param_name, {})
        param_type = param_info.get("type", "str")
        
        # Parse value based on type
        try:
            if param_type == "bool":
                value = value_str.lower() in ['true', '1', 'yes']
            elif param_type == "int":
                value = int(value_str)
            elif param_type == "float":
                value = float(value_str)
            elif param_type == "str":
                if value_str.lower() == "none":
                    value = None
                else:
                    value = value_str
            else:
                value = json.loads(value_str)
        except:
            value = value_str if value_str else None
        
        self.active_gen_extra_args[param_name] = value
        return self.get_gen_extra_list(), self.render_gen_extra_args()
    
    def remove_gen_param(self, param_name):
        """Remove generation param."""
        if param_name in self.active_gen_extra_args:
            self.active_gen_extra_args.pop(param_name)
        return self.get_gen_extra_list(), self.render_gen_extra_args()
    
    def get_init_extra_list(self):
        """Get list of active init params for display."""
        if not self.active_init_extra_args:
            return []
        return [[k, str(v)] for k, v in self.active_init_extra_args.items()]
    
    def get_gen_extra_list(self):
        """Get list of active gen params for display."""
        if not self.active_gen_extra_args:
            return []
        return [[k, str(v)] for k, v in self.active_gen_extra_args.items()]
    
    def render_init_extra_args(self):
        """Render summary of init extra args."""
        if not self.active_init_extra_args:
            return "_No extra init args_"
        return f"**{len(self.active_init_extra_args)} extra init arg(s) active**"
    
    def render_gen_extra_args(self):
        """Render summary of gen extra args."""
        if not self.active_gen_extra_args:
            return "_No extra generation args_"
        return f"**{len(self.active_gen_extra_args)} extra gen arg(s) active**"
    
    # === MODEL MANAGEMENT ===
    def cleanup_model(self):
        """Properly cleanup old model and free memory."""
        if self.model is not None:
            try:
                # Delete model
                del self.model
                self.model = None
                self.current_model_name = None
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache if available
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                print("🧹 Model cleanup completed")
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")
    
    def load_model(self, model_name, dtype, device_map, trust_remote_code):
        """Load model ONCE with init params."""
        if self.current_model_name == model_name and self.model is not None:
            return f"✅ Model '{model_name}' already loaded."
        
        try:
            # CLEANUP OLD MODEL FIRST
            if self.model is not None:
                print(f"🧹 Cleaning up old model: {self.current_model_name}")
                self.cleanup_model()
            
            # Convert string values
            trust_rc = trust_remote_code.lower() in ['true', '1'] if isinstance(trust_remote_code, str) else trust_remote_code
            
            # Build init args with INFRA params
            init_args = {
                "model_id": model_name,
                "dtype": dtype if dtype != "auto" else None,
                "device_map": device_map,
                "trust_remote_code": trust_rc,
            }
            
            # Add extra args (separate INFRA vs RUNTIME)
            infra_extras = {k: v for k, v in self.active_init_extra_args.items() 
                           if k in INFRA_LOAD_PARAMS}
            runtime_extras = {k: v for k, v in self.active_init_extra_args.items() 
                             if k in RUNTIME_FORWARD_PARAMS}
            
            # Merge INFRA params into init_args
            init_args.update(infra_extras)
            
            self.model_config = {
                "model_id": model_name,
                "init_args": init_args,
                "runtime_forward_params": runtime_extras,  # Store separately
                "generation_args": {
                    "max_new_tokens": 1024,
                    "do_sample": False,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 50
                }
            }
            
            # Load new model
            print(f"🔄 Loading new model: {model_name}")
            self.model = HFModel(self.model_config)
            self.model.load_model()
            self.current_model_name = model_name
            
            # Report GPU memory if available
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"📊 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            extra_info = ""
            if infra_extras:
                extra_info += f"\n🔧 Infra extras: {', '.join(infra_extras.keys())}"
            if runtime_extras:
                extra_info += f"\n⚡ Runtime extras: {', '.join(runtime_extras.keys())}"
            
            return f"✅ Model '{model_name}' loaded!{extra_info}"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}"
    
    def update_generation_params(self, max_tokens, temperature, top_p, top_k, do_sample):
        """Update generation params."""
        if not self.model:
            return "⚠️ Load model first!"
        
        try:
            # Convert string values
            do_sample_val = do_sample.lower() in ['true', '1'] if isinstance(do_sample, str) else do_sample
            
            # Build generation args
            gen_args = {
                "max_new_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "do_sample": do_sample_val,
                **self.active_gen_extra_args
            }
            
            # Validate: if do_sample=False, warn about sampling params
            if not do_sample_val and (float(temperature) != 1.0 or float(top_p) != 1.0):
                warning = "\n⚠️ Warning: do_sample=False → temperature, top_p, top_k will be IGNORED"
            else:
                warning = ""
            
            # Update config
            self.model_config["generation_args"] = gen_args
            
            # CRITICAL: Update model's config reference
            self.model.config = self.model_config
            
            # Verify update
            actual_gen_args = self.model.config.get("generation_args", {})
            verification = f"\n✓ Verified: do_sample={actual_gen_args.get('do_sample')}, temp={actual_gen_args.get('temperature')}"
            
            extra_info = ""
            if self.active_gen_extra_args:
                extra_info = f"\n🔧 Extra args: {', '.join(self.active_gen_extra_args.keys())}"
            
            return f"✅ Generation params updated!{warning}{verification}{extra_info}"
            
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
            
            # DEBUG: Print generation args being used
            gen_args = self.model_config.get("generation_args", {})
            print(f"\n🔍 DEBUG Generation Args:")
            print(f"  - do_sample: {gen_args.get('do_sample')}")
            print(f"  - temperature: {gen_args.get('temperature')}")
            print(f"  - top_p: {gen_args.get('top_p')}")
            print(f"  - top_k: {gen_args.get('top_k')}")
            print(f"  - max_new_tokens: {gen_args.get('max_new_tokens')}")
            
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
            
            metadata = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "execution_time": f"{exec_time:.3f}s",
                "generation_time": f"{gen_time:.3f}s",
                "seed": self.seed,
                "model": self.current_model_name,
                "technique": config_str,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "generation_params": gen_args
            }
            
            metadata_display = f"""⏱️ **Execution Time:** {exec_time:.3f}s (Gen: {gen_time:.3f}s)
📊 **Tokens:** Input={input_tokens} | Output={output_tokens}
🎯 **Technique:** {config_str}
🤖 **Model:** {self.current_model_name}
🌱 **Seed:** {self.seed}

⚙️ **Generation Parameters:**
"""
            for k, v in gen_args.items():
                metadata_display += f"• {k}: {v}\n"
            
            self.history.insert(0, {
                "id": len(self.history) + 1,
                "time": metadata["timestamp"],
                "config": config_str,
                "exec": f"{exec_time:.2f}s",
                "text": text
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

# === CUSTOM THEME & CSS ===
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

custom_css = """
/* Container */
.gradio-container {
    max-width: 1600px !important;
    margin: auto;
}

/* Header */
h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 800;
    text-align: center;
    margin: 2rem 0;
    letter-spacing: -0.02em;
}

/* Section headers */
h3 {
    color: #4338ca;
    font-weight: 700;
    font-size: 1.25rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid #e0e7ff;
    padding-bottom: 0.5rem;
}

/* Cards */
.model-setup-card, .gen-params-card {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

/* Primary button */
button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
}

button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
}

/* Secondary button */
button.secondary {
    background: linear-gradient(135deg, #64748b 0%, #475569 100%) !important;
    color: white !important;
}

/* Compact spacing */
.compact-row {
    gap: 0.5rem !important;
}

/* Table styling */
.dataframe {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}

/* Tab styling */
.tab-nav button {
    font-weight: 500;
    border-radius: 8px 8px 0 0;
}

.tab-nav button[aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

/* Improve markdown */
.markdown-text {
    line-height: 1.7;
}

/* Status messages */
.status-success {
    color: #059669;
    font-weight: 600;
}

.status-error {
    color: #dc2626;
    font-weight: 600;
}

/* Add subtle animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.3s ease-out;
}
"""

# === GRADIO UI ===
with gr.Blocks(title="SSA Demo", theme=custom_theme, css=custom_css) as demo:
    gr.Markdown("# 🧠 Structured Sentiment Analysis Demo")
    gr.Markdown("_Advanced NLP inference with configurable prompting techniques_")
    
    gr.Markdown("---")
    
    # === MODEL SETUP & GENERATION PARAMS ===
    with gr.Row(equal_height=False, elem_classes="compact-row"):
        # LEFT: Model Setup
        with gr.Column(scale=1, elem_classes="model-setup-card"):
            gr.Markdown("### 🔧 Model Setup")
            gr.Markdown("_Load once per session_")
            
            model_input = gr.Dropdown(
                choices=DEFAULT_MODEL_SUGGESTIONS,
                value=DEFAULT_MODEL_SUGGESTIONS[0],
                label="Model Name",
                allow_custom_value=True,
                elem_id="model-selector"
            )
            
            dtype_dropdown = gr.Dropdown(
                ["auto", "float16", "bfloat16", "float32"], 
                value="auto", 
                label="dtype"
            )
            
            device_map_input = gr.Textbox(value="auto", label="device_map")
            
            trust_remote_code_dropdown = gr.Dropdown(
                choices=["True", "False"],
                value="True",
                label="trust_remote_code"
            )
            
            # Extra Init Args
            with gr.Accordion("🔧 Extra Init Args", open=False):
                with gr.Row():
                    init_param_search = gr.Dropdown(
                        choices=sorted(list(INIT_PARAM_REGISTRY.keys())),
                        label="Parameter",
                        value=None,
                        scale=2
                    )
                    init_param_value = gr.Textbox(
                        label="Value", 
                        placeholder="e.g., flash_attention_2 or true",
                        scale=2
                    )
                
                with gr.Row():
                    add_init_btn = gr.Button("➕ Add", size="sm", scale=1)
                    remove_init_btn = gr.Button("❌ Remove", size="sm", scale=1)
                
                init_extra_status = gr.Markdown("_No extra init args_")
                
                init_extra_table = gr.Dataframe(
                    headers=["Parameter", "Value"],
                    datatype=["str", "str"],
                    label="Active Extra Init Args",
                    interactive=False,
                    wrap=True
                )
            
            with gr.Row():
                load_model_btn = gr.Button("🔄 Load Model", variant="primary", size="lg", elem_classes="primary")
                cleanup_btn = gr.Button("🧹 Cleanup", variant="stop", size="sm")
            
            model_status = gr.Markdown("_No model loaded_", elem_classes="markdown-text")
        
        # RIGHT: Generation Params
        with gr.Column(scale=1, elem_classes="gen-params-card"):
            gr.Markdown("### ⚙️ Generation Parameters")
            gr.Markdown("_Update anytime without reload_")
            
            with gr.Row():
                max_tokens_slider = gr.Slider(128, 4096, value=1024, step=128, label="max_new_tokens")
                do_sample_dropdown = gr.Dropdown(
                    choices=["True", "False"],
                    value="False",
                    label="do_sample"
                )
            
            with gr.Row():
                temperature_slider = gr.Slider(0.0, 2.0, value=0.1, step=0.05, label="temperature")
                top_p_slider = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="top_p")
                top_k_slider = gr.Slider(1, 100, value=50, step=1, label="top_k")
            
            # Extra Gen Args
            with gr.Accordion("🔧 Extra Generation Args", open=False):
                with gr.Row():
                    gen_param_search = gr.Dropdown(
                        choices=sorted(list(GENERATION_PARAM_REGISTRY.keys())),
                        label="Parameter",
                        value=None,
                        scale=2
                    )
                    gen_param_value = gr.Textbox(
                        label="Value", 
                        placeholder="e.g., 1.1 or 3",
                        scale=2
                    )
                
                with gr.Row():
                    add_gen_btn = gr.Button("➕ Add", size="sm", scale=1)
                    remove_gen_btn = gr.Button("❌ Remove", size="sm", scale=1)
                
                gen_extra_status = gr.Markdown("_No extra generation args_")
                
                gen_extra_table = gr.Dataframe(
                    headers=["Parameter", "Value"],
                    datatype=["str", "str"],
                    label="Active Extra Generation Args",
                    interactive=False,
                    wrap=True
                )
            
            update_gen_btn = gr.Button("🔄 Update Generation Params", variant="secondary", size="lg", elem_classes="secondary")
            gen_status = gr.Markdown("_Default params active_", elem_classes="markdown-text")
    
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
    gr.Markdown("---")
    gr.Markdown("## 🚀 Inference")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="📝 Input Text",
                lines=8,
                placeholder="Enter Vietnamese text for sentiment analysis...",
                value="Tôi rất thích sản phẩm này nhưng nhân viên thái độ tệ quá."
            )
            run_btn = gr.Button("🚀 Run Inference", variant="primary", size="lg", elem_classes="primary")
        
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("🕸️ Visualization"):
                    html_output = gr.HTML(label="SSA Graph")
                
                with gr.TabItem("📄 JSON Output"):
                    json_output = gr.JSON(label="Structured Result")
                
                with gr.TabItem("📥 Full Input"):
                    full_input_output = gr.Textbox(lines=15, max_lines=20, label="Complete Input")
                
                with gr.TabItem("🤖 Raw Response"):
                    raw_response_output = gr.Textbox(lines=15, max_lines=20, label="Model Output")
                
                with gr.TabItem("ℹ️ Metadata"):
                    metadata_output = gr.Markdown(label="Run Metadata")
    
    # === HISTORY ===
    gr.Markdown("---")
    gr.Markdown("## 🕰️ Run History")
    history_table = gr.Dataframe(
        headers=["ID", "Time", "Technique", "Exec", "Input"],
        datatype=["number", "str", "str", "str", "str"],
        elem_classes="dataframe"
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
    
    # Cleanup model
    def cleanup_handler():
        demo_mgr.cleanup_model()
        return "✅ Model cleaned up. Memory freed."
    
    cleanup_btn.click(
        fn=cleanup_handler,
        outputs=[model_status]
    )
    
    # Init extra args
    add_init_btn.click(
        fn=demo_mgr.add_init_param,
        inputs=[init_param_search, init_param_value],
        outputs=[init_extra_table, init_extra_status]
    )
    
    remove_init_btn.click(
        fn=demo_mgr.remove_init_param,
        inputs=[init_param_search],
        outputs=[init_extra_table, init_extra_status]
    )
    
    # Gen extra args
    add_gen_btn.click(
        fn=demo_mgr.add_gen_param,
        inputs=[gen_param_search, gen_param_value],
        outputs=[gen_extra_table, gen_extra_status]
    )
    
    remove_gen_btn.click(
        fn=demo_mgr.remove_gen_param,
        inputs=[gen_param_search],
        outputs=[gen_extra_table, gen_extra_status]
    )
    
    # Model & generation
    load_model_btn.click(
        fn=demo_mgr.load_model,
        inputs=[model_input, dtype_dropdown, device_map_input, trust_remote_code_dropdown],
        outputs=[model_status]
    )
    
    update_gen_btn.click(
        fn=demo_mgr.update_generation_params,
        inputs=[max_tokens_slider, temperature_slider, top_p_slider, top_k_slider, do_sample_dropdown],
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
## 💡 Quick Guide

**Model Setup:**
- Select model from dropdown or enter custom HuggingFace model ID
- Configure device, dtype, and attention implementation
- Add extra init args for advanced control (quantization, memory optimization)
- Load model once and reuse throughout session

**Generation Parameters:**
- Adjust sampling strategy with `do_sample` toggle
- Fine-tune temperature, top-p, top-k for output control
- **Note:** Sampling params only work when `do_sample=True`
- Update parameters anytime without reloading model

**Extra Args Format:**
- Boolean: `true` / `false`
- Numbers: `1.1` / `50`  
- Strings: `flash_attention_2` / `auto`
- None: `none`

**Prompting Techniques:**
- Few-shot: Learn from examples
- Chain-of-Thought: Step-by-step reasoning
- Plan-and-Solve: Strategic decomposition
- Re-reading: Enhanced context processing

**Tips:**
- 🔧 Use extra args for model-specific optimizations
- 📊 Check metadata tab for detailed performance metrics
- 🧹 Cleanup model before loading a new one to free memory
- 🕰️ View history to compare different configurations

Built with ❤️ using Transformers & Gradio
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)