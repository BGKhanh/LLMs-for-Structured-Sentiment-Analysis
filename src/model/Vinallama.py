# src/model/Vinallama.py

import re
import json
import time
import torch
from typing import Tuple, Dict, Any, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from .BaseModel import BaseModel


class VinallamaModel(BaseModel):
    """
    VinaLlama 7B Chat implementation.
    
    Handles:
    - VinaLlama specific loading (Based on LlamaForCausalLM)
    - Standard chat template application (system/user roles)
    - Batch inference with precise token tracking
    - JSON response extraction
    
    Reference: vilm/vinallama-7b-chat
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VinaLlama model.
        
        Args:
            config: Configuration dict or ModelConfig object
        """
        super().__init__(config)
        self.model_class = AutoModelForCausalLM
        self.tokenizer = None
        self.chat_template_builder = self._build_chat_messages

    def load_model(self) -> None:
        """Load VinaLlama model and tokenizer."""
        if self.is_loaded:
            print("✅ Model already loaded!")
            return
        
        # Support both dict and ModelConfig object
        if isinstance(self.config, dict):
            # Legacy/Dict support
            init_args = self.config.get("init_args", {})
            if hasattr(init_args, "to_dict"):
                kwargs = init_args.to_dict()
            else:
                kwargs = init_args
        else:
            # ModelConfig object support (Preferred)
            kwargs = self.config.init_args.to_dict()
            
        model_id = kwargs.pop("model_id") # Extract ID
        
        # Convert dtype string to actual torch type if needed
        if kwargs.get("dtype") != "auto" and isinstance(kwargs.get("dtype"), str):
            kwargs["dtype"] = getattr(torch, kwargs["dtype"])
        
        print(f"🔧 Loading VinaLlama model: {model_id}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=kwargs.get("trust_remote_code", True)
            )
            
            # Configure pad_token if missing (Critical for batch processing)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                print("  ⚙️  Configured pad_token_id = eos_token_id")
            
            # Set padding side to left for decoder-only models
            self.tokenizer.padding_side = "left"
            
            # Load model
            self.model = self.model_class.from_pretrained(
                model_id,
                **kwargs
            ).eval()
            
            self.is_loaded = True
            print("✅ VinaLlama model and tokenizer loaded successfully!")
            
            if hasattr(self.model, 'hf_device_map'):
                print(f"📍 Device map: {self.model.hf_device_map}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load VinaLlama model: {str(e)}")

    def _build_chat_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        """Return chat messages in standard format."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
    def generate_single(self, system_prompt: str, user_prompt: str) -> Tuple[str, float]:
        """Generate response for single input."""
        if not self.is_loaded:
            self.load_model()
        
        messages = self.chat_template_builder(system_prompt, user_prompt)
                
        try:
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            input_len = model_inputs["input_ids"].shape[-1]
            
            # Get generation kwargs
            if hasattr(self.config, 'get_generation_kwargs'):
                gen_kwargs = self.config.get_generation_kwargs()
            else:
                gen_kwargs = {
                    "max_new_tokens": self.config.get("max_tokens", 512),
                    "do_sample": self.config.get("do_sample", True),
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                if self.config.get("do_sample"):
                    gen_kwargs["temperature"] = self.config.get("temperature", 0.6)
            
            # Generate
            start_time = time.time()
            with torch.inference_mode(), torch.autocast(device_type="cuda"):
                generated_ids = self.model.generate(
                    **model_inputs,
                    **gen_kwargs
                )
            
            generation_time = time.time() - start_time
            
            # Extract output tokens only
            output_ids = generated_ids[0][input_len:]
            
            # Decode
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            return response, generation_time
            
        except Exception as e:
            print(f"❌ Error in generate_single: {str(e)}")
            return "{}", 0.0

    def generate_batch(
        self, 
        inputs: Union[List[List[Dict[str, Any]]], Dict[str, torch.Tensor]]
    ) -> Tuple[List[str], float, List[int]]:
        """
        Generate responses for batch inputs.
        
        Args:
            inputs: Either:
                - Dict[str, torch.Tensor]: Tokenized inputs from collator
                - List[List[Dict]]: Raw messages for manual batching
        
        Returns:
            Tuple of (batch_responses, generation_time, output_token_counts)
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Get generation kwargs
            if isinstance(self.config, dict):
                gen_args = self.config.get("generation_args", {})
                if hasattr(gen_args, "to_dict"):
                    gen_kwargs = gen_args.to_dict()
                else:
                    gen_kwargs = gen_args
            else:
                gen_kwargs = self.config.generation_args.to_dict()
                
            # This parameter is only use by tokenizer.apply_chat_template()   
            enable_thinking = gen_kwargs.pop("enable_thinking", False)
            
            start_time = time.time()
            
            # ===== MODE DETECTION =====
            if isinstance(inputs, dict):
                # Mode 1: Tokenized inputs from DataLoader
                tokenized_inputs = {
                    k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                    if k in ['input_ids', 'attention_mask']
                }
                input_length = tokenized_inputs['input_ids'].shape[1]
                
            else:
                # Mode 2: Raw messages (manual batching)
                texts = []
                for msgs in inputs:
                    text = self.tokenizer.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking
                    )
                    texts.append(text)
                
                # Tokenize batch
                tokenized_inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True
                ).to(self.model.device)
                input_length = tokenized_inputs['input_ids'].shape[1]
            
            # ===== BATCH GENERATION =====
            with torch.inference_mode(), torch.autocast(device_type="cuda"):
                generated_outputs = self.model.generate(
                    **tokenized_inputs,
                    **gen_kwargs
                )
            
            batch_time = time.time() - start_time
            
            # ===== DECODE OUTPUTS & COUNT TOKENS =====
            output_ids = generated_outputs[:, input_length:]
            
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
        
            output_token_counts = []
            for i in range(output_ids.size(0)):
                seq = output_ids[i]
                count = 0
                for token_id in seq:
                    tid = token_id.item()
                    if (pad_token_id is not None and tid == pad_token_id) or \
                       (eos_token_id is not None and tid == eos_token_id):
                        break
                    count += 1
                output_token_counts.append(count)
                
            batch_responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            # Trim whitespace
            batch_responses = [r.strip() for r in batch_responses]
            
            return batch_responses, batch_time, output_token_counts
            
        except Exception as e:
            print(f"❌ Error in generate_batch: {str(e)}")
            # Return empty responses based on input type
            if isinstance(inputs, dict):
                batch_size = inputs['input_ids'].shape[0]
            else:
                batch_size = len(inputs)
            return ["{}"] * batch_size, 0.0, [0] * batch_size

    def extract_response(self, raw_response: str) -> str:
        """
        Extract JSON from VinaLlama's response.
        """
        # Step 1: Try to extract from ... ``` block
        match = re.search(r'\s*([\s\S]*?)\s*```', raw_response, re.DOTALL)
        if match:
            json_candidate = match.group(1).strip()
            try:
                json.loads(json_candidate)
                return json_candidate
            except json.JSONDecodeError:
                pass
        
        # Step 2: Find first valid JSON object using brace matching
        first_brace = raw_response.find('{')
        if first_brace != -1:
            open_braces = 0
            for i in range(first_brace, len(raw_response)):
                if raw_response[i] == '{':
                    open_braces += 1
                elif raw_response[i] == '}':
                    open_braces -= 1
                    if open_braces == 0:
                        json_candidate = raw_response[first_brace:i+1]
                        try:
                            json.loads(json_candidate)
                            return json_candidate.strip()
                        except json.JSONDecodeError:
                            break
                            
        # Step 3: Fallback - clean markdown
        return raw_response.replace('', '').replace('```', '').strip()