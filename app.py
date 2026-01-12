# app_streamlit.py

import streamlit as st
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path
import gc

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
INFRA_LOAD_PARAMS = {
    "cache_dir": {"default": None, "type": "str", "group": "Cache"},
    "force_download": {"default": False, "type": "bool", "group": "Cache"},
    "local_files_only": {"default": False, "type": "bool", "group": "Cache"},
    "token": {"default": None, "type": "str", "group": "Auth"},
    "revision": {"default": "main", "type": "str", "group": "Version"},
    "use_safetensors": {"default": None, "type": "bool", "group": "Loading"},
    "attn_implementation": {
        "default": None, "type": "str", "group": "Performance",
        "choices": ["eager", "sdpa", "flash_attention_2", "flash_attention_3"]
    },
    "max_memory": {"default": None, "type": "dict", "group": "Device"},
    "offload_folder": {"default": None, "type": "str", "group": "Memory"},
    "offload_buffers": {"default": False, "type": "bool", "group": "Memory"},
    "quantization_config": {"default": None, "type": "dict", "group": "Quantization"},
    "load_in_4bit": {"default": False, "type": "bool", "group": "Quantization"},
    "load_in_8bit": {"default": False, "type": "bool", "group": "Quantization"},
    "subfolder": {"default": "", "type": "str", "group": "Loading"},
    "variant": {"default": None, "type": "str", "group": "Loading"},
    "low_cpu_mem_usage": {"default": False, "type": "bool", "group": "Memory"},
}

RUNTIME_FORWARD_PARAMS = {
    "use_cache": {"default": True, "type": "bool"},
    "output_hidden_states": {"default": False, "type": "bool"},
    "output_attentions": {"default": False, "type": "bool"},
    "return_dict": {"default": True, "type": "bool"},
}

INIT_PARAM_REGISTRY = {**INFRA_LOAD_PARAMS, **RUNTIME_FORWARD_PARAMS}

# GENERATION PARAMS
GENERATION_PARAM_REGISTRY = {}
try:
    gen_config = GenerationConfig()
    for name, value in gen_config.to_dict().items():
        if name not in ['transformers_version', '_from_model_config']:
            GENERATION_PARAM_REGISTRY[name] = {
                "default": value,
                "type": type(value).__name__
            }
except:
    pass

# === CONFIGS ===
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

# === PAGE CONFIG ===
st.set_page_config(
    page_title="SSA Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === INITIALIZE SESSION STATE ===
def init_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.seed = 42
        setup_reproducible_environment(seed=42)
        
        st.session_state.model = None
        st.session_state.current_model_name = None
        st.session_state.model_config = None
        st.session_state.prompt_template = None
        st.session_state.current_prompt_config = None
        st.session_state.history = []
        
        st.session_state.active_init_extra_args = {}
        st.session_state.active_gen_extra_args = {}
        
        st.session_state.initialized = True

init_session_state()

# === HELPER FUNCTIONS ===
def cleanup_model():
    """Cleanup old model and free memory."""
    if st.session_state.model is not None:
        try:
            del st.session_state.model
            st.session_state.model = None
            st.session_state.current_model_name = None
            
            gc.collect()
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            st.success("🧹 Model cleaned up successfully")
        except Exception as e:
            st.warning(f"⚠️ Cleanup warning: {e}")

def load_model(model_name, dtype, device_map, trust_remote_code):
    """Load model with init params."""
    if st.session_state.current_model_name == model_name and st.session_state.model is not None:
        st.info(f"✅ Model '{model_name}' already loaded")
        return
    
    try:
        # Cleanup old model
        if st.session_state.model is not None:
            with st.spinner(f"🧹 Cleaning up old model: {st.session_state.current_model_name}"):
                cleanup_model()
        
        # Build init args
        init_args = {
            "model_id": model_name,
            "dtype": dtype if dtype != "auto" else None,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        
        # Add extra args
        infra_extras = {k: v for k, v in st.session_state.active_init_extra_args.items() 
                       if k in INFRA_LOAD_PARAMS}
        runtime_extras = {k: v for k, v in st.session_state.active_init_extra_args.items() 
                         if k in RUNTIME_FORWARD_PARAMS}
        
        init_args.update(infra_extras)
        
        st.session_state.model_config = {
            "model_id": model_name,
            "init_args": init_args,
            "runtime_forward_params": runtime_extras,
            "generation_args": {
                "max_new_tokens": 1024,
                "do_sample": False,
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 50
            }
        }
        
        # Load model
        with st.spinner(f"🔄 Loading model: {model_name}..."):
            st.session_state.model = HFModel(st.session_state.model_config)
            st.session_state.model.load_model()
            st.session_state.current_model_name = model_name
        
        # Report GPU memory
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            st.info(f"📊 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        st.success(f"✅ Model '{model_name}' loaded successfully!")
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())

def update_generation_params(max_tokens, temperature, top_p, top_k, do_sample):
    """Update generation params."""
    if st.session_state.model is None:
        st.warning("⚠️ Load model first!")
        return
    
    try:
        gen_args = {
            "max_new_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "do_sample": do_sample,
            **st.session_state.active_gen_extra_args
        }
        
        # Validate
        if not do_sample and (temperature != 1.0 or top_p != 1.0):
            st.warning("⚠️ do_sample=False → temperature, top_p, top_k will be IGNORED")
        
        st.session_state.model_config["generation_args"] = gen_args
        st.session_state.model.config = st.session_state.model_config
        
        st.success(f"✅ Generation params updated! do_sample={do_sample}, temp={temperature}")
        
    except Exception as e:
        st.error(f"❌ Error: {e}")

def load_prompt_template(technique, n_shot, examples_pool, plus_mode, re_add_method):
    """Load prompt template."""
    config_key = f"{technique}_{n_shot}_{examples_pool}_{plus_mode}_{re_add_method}"
    
    if st.session_state.current_prompt_config == config_key and st.session_state.prompt_template:
        return True
    
    try:
        if technique == "few_shot":
            if n_shot > 0 and not Path(examples_pool).exists():
                st.error(f"⚠️ Pool not found: {examples_pool}")
                return False
            st.session_state.prompt_template = FewShotPrompt(
                eng=False, n_shot=n_shot,
                examples_pool_path=examples_pool if n_shot > 0 else None
            )
            
        elif technique == "few_shot_cot":
            st.session_state.prompt_template = FewShotCoTPrompt(eng=False, n_shot=n_shot)
            
        elif technique == "plan_and_solve":
            st.session_state.prompt_template = PlanAndSolvePrompt(
                eng=False, plus=plus_mode,
                n_shot=n_shot if plus_mode else 0
            )
            
        elif technique == "re_reading":
            needs_pool = re_add_method == "FewShot"
            if needs_pool and n_shot > 0 and not Path(examples_pool).exists():
                st.error(f"⚠️ Pool not found: {examples_pool}")
                return False
            st.session_state.prompt_template = ReReadingPrompt(
                eng=False, add_method=re_add_method, n_shot=n_shot,
                examples_pool_path=examples_pool if needs_pool and n_shot > 0 else None
            )
            
        elif technique == "re2_pas_cot":
            st.session_state.prompt_template = Re2PaSCoTPrompt(eng=False, n_shot=n_shot)
        
        st.session_state.prompt_template.prepare()
        st.session_state.current_prompt_config = config_key
        return True
        
    except Exception as e:
        st.error(f"❌ Error loading prompt: {e}")
        return False

def run_inference(text, technique, n_shot, examples_pool, plus_mode, re_add_method):
    """Run inference."""
    if st.session_state.model is None:
        st.error("⚠️ Load model first!")
        return None
    
    if not load_prompt_template(technique, n_shot, examples_pool, plus_mode, re_add_method):
        return None
    
    try:
        with st.spinner("🚀 Running inference..."):
            start = time.time()
            system_prompt, user_prompt = st.session_state.prompt_template.get_prompt(
                text, f"demo_{len(st.session_state.history)+1}"
            )
            
            # DEBUG
            gen_args = st.session_state.model_config.get("generation_args", {})
            st.write("🔍 **DEBUG Generation Args:**")
            st.json(gen_args)
            
            raw_response, gen_time = st.session_state.model.generate_single(system_prompt, user_prompt)
            json_str = st.session_state.model.extract_response(raw_response)
            processed = postprocess_response(json_str, text, f"demo_{len(st.session_state.history)+1}")
            result = json.loads(processed)
            exec_time = time.time() - start
            
            html_graph = convert_ssa_to_spacy(text, result)
            
            config_str = PROMPT_TECHNIQUES[technique]['name']
            if n_shot > 0:
                config_str += f" ({n_shot}-shot)"
            
            input_tokens = len(system_prompt.split()) + len(user_prompt.split())
            output_tokens = len(raw_response.split())
            
            # Save to history
            st.session_state.history.insert(0, {
                "id": len(st.session_state.history) + 1,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "technique": config_str,
                "exec_time": f"{exec_time:.2f}s",
                "gen_time": f"{gen_time:.2f}s",
                "text": text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": raw_response,
                "result": result,
                "html_graph": html_graph,
                "gen_args": gen_args
            })
            
            return st.session_state.history[0]
            
    except Exception as e:
        st.error(f"❌ Inference error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# === MAIN APP ===
st.title("🧠 Structured Sentiment Analysis Demo")

# === SIDEBAR: MODEL & PARAMS ===
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model Setup
    with st.expander("🔧 Model Setup", expanded=True):
        model_name = st.selectbox(
            "Model Name",
            options=DEFAULT_MODEL_SUGGESTIONS,
            index=0
        )
        
        col1, col2 = st.columns(2)
        with col1:
            dtype = st.selectbox("dtype", ["auto", "float16", "bfloat16", "float32"])
            device_map = st.text_input("device_map", value="auto")
        with col2:
            trust_remote_code = st.selectbox("trust_remote_code", ["True", "False"], index=0)
        
        st.markdown("**Extra Init Args**")
        init_param = st.selectbox("Parameter", [""] + sorted(list(INIT_PARAM_REGISTRY.keys())))
        init_value = st.text_input("Value", key="init_value")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add Init", use_container_width=True):
                if init_param and init_value:
                    try:
                        param_info = INIT_PARAM_REGISTRY.get(init_param, {})
                        param_type = param_info.get("type", "str")
                        if param_type == "bool":
                            value = init_value.lower() in ['true', '1', 'yes']
                        elif param_type == "int":
                            value = int(init_value)
                        elif param_type == "float":
                            value = float(init_value)
                        else:
                            value = init_value
                        st.session_state.active_init_extra_args[init_param] = value
                        st.success(f"Added {init_param}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        with col2:
            if st.button("❌ Remove Init", use_container_width=True):
                if init_param in st.session_state.active_init_extra_args:
                    del st.session_state.active_init_extra_args[init_param]
                    st.success(f"Removed {init_param}")
        
        if st.session_state.active_init_extra_args:
            st.dataframe(
                [[k, str(v)] for k, v in st.session_state.active_init_extra_args.items()],
                column_config={"0": "Parameter", "1": "Value"},
                hide_index=True,
                use_container_width=True
            )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Load Model", type="primary", use_container_width=True):
                load_model(model_name, dtype, device_map, trust_remote_code == "True")
        with col2:
            if st.button("🧹 Cleanup", use_container_width=True):
                cleanup_model()
    
    # Generation Params
    with st.expander("⚙️ Generation Parameters", expanded=True):
        max_tokens = st.slider("max_new_tokens", 128, 4096, 1024, step=128)
        temperature = st.slider("temperature", 0.0, 2.0, 0.1, step=0.05)
        top_p = st.slider("top_p", 0.0, 1.0, 0.9, step=0.05)
        top_k = st.slider("top_k", 1, 100, 50)
        do_sample = st.selectbox("do_sample", ["False", "True"], index=0) == "True"
        
        st.markdown("**Extra Gen Args**")
        gen_param = st.selectbox("Parameter", [""] + sorted(list(GENERATION_PARAM_REGISTRY.keys())))
        gen_value = st.text_input("Value", key="gen_value")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add Gen", use_container_width=True):
                if gen_param and gen_value:
                    try:
                        param_info = GENERATION_PARAM_REGISTRY.get(gen_param, {})
                        param_type = param_info.get("type", "str")
                        if param_type == "bool":
                            value = gen_value.lower() in ['true', '1', 'yes']
                        elif param_type == "int":
                            value = int(gen_value)
                        elif param_type == "float":
                            value = float(gen_value)
                        else:
                            value = gen_value
                        st.session_state.active_gen_extra_args[gen_param] = value
                        st.success(f"Added {gen_param}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        with col2:
            if st.button("❌ Remove Gen", use_container_width=True):
                if gen_param in st.session_state.active_gen_extra_args:
                    del st.session_state.active_gen_extra_args[gen_param]
                    st.success(f"Removed {gen_param}")
        
        if st.session_state.active_gen_extra_args:
            st.dataframe(
                [[k, str(v)] for k, v in st.session_state.active_gen_extra_args.items()],
                column_config={"0": "Parameter", "1": "Value"},
                hide_index=True,
                use_container_width=True
            )
        
        if st.button("🔄 Update Params", type="secondary", use_container_width=True):
            update_generation_params(max_tokens, temperature, top_p, top_k, do_sample)
    
    # Prompt Technique
    with st.expander("🎯 Prompt Technique", expanded=False):
        technique = st.selectbox("Technique", list(PROMPT_TECHNIQUES.keys()))
        
        params = PROMPT_TECHNIQUES[technique]["params"]
        
        n_shot = 3
        if "n_shot" in params or "n_shot_conditional" in params:
            n_shot = st.slider("N-shot", 0, 5, 3)
        
        examples_pool = EXAMPLES_POOLS[0]
        if "examples_pool" in params or "examples_pool_conditional" in params:
            examples_pool = st.selectbox("Examples Pool", EXAMPLES_POOLS)
        
        plus_mode = True
        if "plus_mode" in params:
            plus_mode = st.checkbox("PS+ Mode", value=True)
        
        re_add_method = "none"
        if "add_method" in params:
            re_add_method = st.selectbox("Add Method", ["none", "0_CoT", "FewShot", "FewShot_CoT", "PaS"])

# === MAIN AREA: INPUT & RESULTS ===
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Input")
    input_text = st.text_area(
        "Text",
        value="Tôi rất thích sản phẩm này nhưng nhân viên thái độ tệ quá.",
        height=200
    )
    
    if st.button("🚀 Run Inference", type="primary", use_container_width=True):
        result = run_inference(input_text, technique, n_shot, examples_pool, plus_mode, re_add_method)
        if result:
            st.session_state.current_result = result

with col2:
    st.subheader("📊 Results")
    
    if 'current_result' in st.session_state and st.session_state.current_result:
        result = st.session_state.current_result
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🕸️ Visualization", "📄 JSON", "📥 Full Input", "🤖 Raw Response", "ℹ️ Metadata"])
        
        with tab1:
            st.components.v1.html(result['html_graph'], height=600, scrolling=True)
        
        with tab2:
            st.json(result['result'])
        
        with tab3:
            st.markdown("**System Prompt:**")
            st.text(result['system_prompt'])
            st.markdown("**User Prompt:**")
            st.text(result['user_prompt'])
        
        with tab4:
            st.text(result['raw_response'])
        
        with tab5:
            st.markdown(f"""
**Execution Time:** {result['exec_time']} (Gen: {result['gen_time']})  
**Tokens:** Input={result['input_tokens']} | Output={result['output_tokens']}  
**Technique:** {result['technique']}  
**Model:** {st.session_state.current_model_name}  
**Seed:** {st.session_state.seed}

**Generation Parameters:**
""")
            st.json(result['gen_args'])

# === HISTORY ===
st.divider()
st.subheader("🕰️ Run History")

if st.session_state.history:
    history_data = [[
        h['id'],
        h['timestamp'],
        h['technique'],
        h['exec_time'],
        h['text'][:40] + "..."
    ] for h in st.session_state.history]
    
    st.dataframe(
        history_data,
        column_config={
            "0": "ID",
            "1": "Timestamp",
            "2": "Technique",
            "3": "Exec Time",
            "4": "Input"
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.info("No runs yet. Run inference to see history.")

# === FOOTER ===
st.divider()
st.markdown("""
**💡 Tips:**
- Load model once in sidebar, update generation params anytime
- Extra args: Select parameter → Enter value → Click Add
- All runs are saved in history for comparison
- Seed is fixed at 42 for reproducibility
""")

if __name__ == "__main__":
    pass  # Streamlit runs automatically