# src/model/HFModel.py

import time
import torch
from typing import Tuple, Dict, Any, List, Union, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.postprocessing import extract_json_from_response
import gc

class HFModel:
    """
    Unified HuggingFace Model Wrapper.
    
    Generic implementation for any AutoModelForCausalLM model.
    Designed for flexibility in demo/testing scenarios with dynamic parameter updates.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HFModel.
        
        Args:
            config: Configuration dictionary with structure:
                {
                    "model_id": "model-name" or
                    "init_args": {
                        "model_id": "model-name",
                        "dtype": "auto"/"float16"/"bfloat16",
                        "device_map": "auto"/"cuda"/"cpu",
                        "trust_remote_code": True/False
                    },
                    "generation_args": {
                        "max_new_tokens": 1024,
                        "do_sample": False,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        ...
                    }
                }
        """
        self.config = config
        
        # === Extract and store init_args ===
        if "init_args" in config:
            self.init_args = config["init_args"].copy()
        else:
            # Flat config format
            self.init_args = {
                "model_id": config.get("model_id"),
                "dtype": config.get("dtype", "auto"),
                "device_map": config.get("device_map", "auto"),
                "trust_remote_code": config.get("trust_remote_code", True)
            }
        
        self.model_id = self.init_args.get("model_id")
        if not self.model_id:
            raise ValueError("model_id is required in config")
        
        # === Extract and store generation_args ===
        if "generation_args" in config:
            self.generation_args = config["generation_args"].copy()
        else:
            # Default generation args
            self.generation_args = {
                "max_new_tokens": config.get("max_new_tokens", 1024),
                "do_sample": config.get("do_sample", False),
                "temperature": config.get("temperature", 0.7),
                "top_p": config.get("top_p", 0.9)
            }
        
        # Model & Tokenizer (loaded lazily)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Chat template builder (will be set during load)
        self.chat_template_builder = self._build_chat_messages

    def _build_chat_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        """
        Build chat messages in standard format.
        
        Override this method or attribute for model-specific formats.
        Default: Flat format (works for Llama, Qwen, Mistral, etc.)
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def load_model(self) -> None:
        """Load model and tokenizer using Auto classes."""
        if self.is_loaded:
            print(f"✅ Model {self.model_id} already loaded!")
            return

        print(f"🔧 Loading HFModel: {self.model_id}")
        
        # Prepare loading kwargs
        trust_remote_code = self.init_args.get("trust_remote_code", True)
        device_map = self.init_args.get("device_map", "auto")
        
        # Handle dtype conversion
        dtype_val = self.init_args.get("dtype", "auto")
        if isinstance(dtype_val, str):
            if dtype_val == "auto" or dtype_val is None or dtype_val == "None":
                dtype = None  # Let transformers auto-detect
            elif hasattr(torch, dtype_val):
                dtype = getattr(torch, dtype_val)
            else:
                dtype = None
        else:
            dtype = dtype_val  # Already a torch.dtype object

        try:
            # === 1. Load Tokenizer ===
            print("  📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=trust_remote_code
            )
            
            # Set padding side (standard for decoder-only models)
            self.tokenizer.padding_side = "left"

            # === 2. Load Model ===
            print("  📥 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=device_map,
                dtype=dtype,
                trust_remote_code=trust_remote_code
            ).eval()

            # === 3. Configure pad_token_id (Critical for batch generation) ===
            print("  ⚙️  Configuring pad_token_id...")
            
            # Step 1: Check if model.generation_config.pad_token_id exists
            if self.model.generation_config.pad_token_id is not None:
                print(f"    ✓ model.generation_config.pad_token_id already set: {self.model.generation_config.pad_token_id}")
            else:
                # Step 2: Check tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None:
                    # Use tokenizer's pad_token_id
                    self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
                    print(f"    ✓ Set model.generation_config.pad_token_id = tokenizer.pad_token_id ({self.tokenizer.pad_token_id})")
                else:
                    # Step 3: Fallback - use tokenizer.pad_token string and map via added_tokens_decoder
                    if self.tokenizer.pad_token is not None:
                        # Find token ID from added_tokens_decoder or vocab
                        pad_token_str = self.tokenizer.pad_token
                        
                        # Try to get from vocab
                        if hasattr(self.tokenizer, 'get_vocab'):
                            vocab = self.tokenizer.get_vocab()
                            if pad_token_str in vocab:
                                pad_token_id = vocab[pad_token_str]
                                self.tokenizer.pad_token_id = pad_token_id
                                self.model.generation_config.pad_token_id = pad_token_id
                                print(f"    ✓ Mapped pad_token '{pad_token_str}' to ID {pad_token_id}")
                        else:
                            # Last resort: use eos_token_id
                            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
                            print(f"    ⚠️  Fallback: Set pad_token_id = eos_token_id ({self.tokenizer.eos_token_id})")
                    else:
                        # Ultimate fallback
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
                        print(f"    ⚠️  Fallback: Set pad_token_id = eos_token_id ({self.tokenizer.eos_token_id})")
            
            self.is_loaded = True
            print(f"✅ Loaded {self.model_id} successfully!")
            
            if hasattr(self.model, 'hf_device_map'):
                print(f"📍 Device map: {self.model.hf_device_map}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_id}: {str(e)}")

    def update_generation_args(self, **kwargs):
        """
        Update generation arguments dynamically.
        
        Useful for demo scenarios where parameters change frequently.
        
        Example:
            model.update_generation_args(temperature=0.9, top_p=0.95)
        """
        self.generation_args.update(kwargs)
        print(f"🔄 Updated generation_args: {kwargs}")
        
    def _get_autocast_dtype(self) -> Union[torch.dtype, None]:
        """Helper to get torch dtype from config for autocast."""
        dtype_str = self.init_args.get("dtype", "auto")
        if dtype_str != "auto" and hasattr(torch, dtype_str):
            return getattr(torch, dtype_str)
        return None
    
    def generate_single(self, system_prompt: str, user_prompt: str, **kwargs) -> Tuple[str, float]:
        """
        Generate response for single input.
        
        Args:
            system_prompt: System instruction
            user_prompt: User input
            **kwargs: Temporary override for generation parameters (doesn't modify self.generation_args)
        
        Returns:
            Tuple of (response_text, generation_time)
        """
        if not self.is_loaded:
            self.load_model()

        # Build messages using chat template builder
        messages = self.chat_template_builder(system_prompt, user_prompt)

        try:
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]

            # Merge generation_args with runtime overrides
            gen_kwargs = self.generation_args.copy()
            gen_kwargs.update(kwargs)
            
            # Clean up non-generation params
            gen_kwargs.pop("enable_thinking", None)
            
            # Ensure pad_token_id is set
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            
            start_time = time.time()
            with torch.inference_mode(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=self._get_autocast_dtype()):
                outputs = self.model.generate(**inputs, **gen_kwargs)
            generation_time = time.time() - start_time
            
            # Decode only new tokens
            output_ids = outputs[0][input_len:]
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            return response, generation_time

        except Exception as e:
            print(f"❌ Error in generate_single: {e}")
            return "{}", 0.0

    def generate_batch(
        self, 
        inputs: Union[List[List[Dict[str, Any]]], Dict[str, torch.Tensor]], 
        **kwargs
    ) -> Tuple[List[str], float, List[int]]:
        """
        Batch inference supporting both raw messages and tokenized inputs.
        
        Args:
            inputs: Either:
                - List[List[Dict]]: Raw message lists for chat template
                - Dict[str, torch.Tensor]: Pre-tokenized inputs from DataLoader
            **kwargs: Temporary override for generation parameters
        
        Returns:
            Tuple of (responses, generation_time, output_token_counts)
        """
        if not self.is_loaded:
            self.load_model()
            
        try:
            # === MODE DETECTION & TOKENIZATION ===
            if isinstance(inputs, dict):
                # Mode 1: Pre-tokenized inputs (from DataLoader)
                tokenized_inputs = {
                    k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                    if k in ['input_ids', 'attention_mask']
                }
                input_length = tokenized_inputs['input_ids'].shape[1]
            else:
                # Mode 2: Raw messages list
                texts = [
                    self.tokenizer.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=True
                    )
                    for msg in inputs
                ]
                tokenized_inputs = self.tokenizer(
                    texts, return_tensors="pt", padding=True
                ).to(self.model.device)
                input_length = tokenized_inputs['input_ids'].shape[1]

            # === PREPARE GENERATION KWARGS ===
            gen_kwargs = self.generation_args.copy()
            gen_kwargs.update(kwargs)
            gen_kwargs.pop("enable_thinking", None)
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            
            # === GENERATE ===
            start_time = time.time()
            with torch.inference_mode(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=self._get_autocast_dtype()):
                outputs = self.model.generate(**tokenized_inputs, **gen_kwargs)
            generation_time = time.time() - start_time
            
            # === PROCESS OUTPUTS ===
            output_ids = outputs[:, input_length:]
            
            # Count output tokens (exclude PAD and EOS)
            output_token_counts = []
            pad_id = self.tokenizer.pad_token_id
            eos_id = self.tokenizer.eos_token_id
            
            for seq in output_ids:
                count = 0
                for tid in seq:
                    tid_val = tid.item()
                    if (pad_id is not None and tid_val == pad_id) or \
                       (eos_id is not None and tid_val == eos_id):
                        break
                    count += 1
                output_token_counts.append(count)
            
            # Decode responses
            responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            responses = [r.strip() for r in responses]
            
            return responses, generation_time, output_token_counts

        except Exception as e:
            print(f"❌ Error in generate_batch: {e}")
            batch_size = len(inputs) if isinstance(inputs, list) else inputs['input_ids'].shape[0]
            return ["{}"] * batch_size, 0.0, [0] * batch_size

    def extract_response(self, raw_response: str) -> str:
        """
        Extract JSON from raw model response.
        
        Uses centralized postprocessing utility for consistency.
        """
        return extract_json_from_response(raw_response)

    def cleanup(self):
        """Clean up GPU memory."""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            
            # Aggressive cleanup for multi-GPU
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        
        print("🧹 Model cleaned up")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for debugging."""
        return {
            "model_id": self.model_id,
            "is_loaded": self.is_loaded,
            "dtype": str(self.init_args.get("dtype")),
            "device_map": self.init_args.get("device_map"),
            "generation_args": self.generation_args
        }