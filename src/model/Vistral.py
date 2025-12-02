# src/model/Vistral.py

import re
import json
import time
import torch
from typing import Tuple, Dict, Any, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from .BaseModel import BaseModel


class VistralModel(BaseModel):
    """
    Viet-Mistral/Vistral-7B-Chat implementation.
    
    Handles:
    - Vistral-specific loading (AutoModelForCausalLM)
    - Chat template application (system/user/assistant roles)
    - Batch inference with token tracking
    - JSON response extraction
    
    Reference: https://huggingface.co/Viet-Mistral/Vistral-7B-Chat
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Vistral model.
        
        Args:
            config: Configuration dict or ModelConfig object
        """
        super().__init__(config)
        self.model_class = AutoModelForCausalLM
        self.tokenizer = None
        
        self.enable_thinking = self.config.get("enable_thinking", False) if isinstance(self.config, dict) else getattr(self.config, "enable_thinking", False)
        self.chat_template_builder = self._build_chat_messages

    def load_model(self) -> None:
        """Load Vistral model and tokenizer."""
        if self.is_loaded:
            print("✅ Model already loaded!")
            return
        
        # Support both dict and ModelConfig object
        if hasattr(self.config, 'model_id'):
            # ModelConfig object
            model_id = self.config.model_id
            pretrained_kwargs = self.config.get_from_pretrained_kwargs()
        else:
            # Dict (backward compatible)
            model_id = self.config["model_id"]
            pretrained_kwargs = {
                "device_map": self.config.get("device_map", "auto"),
                "torch_dtype": self.config.get("torch_dtype", "auto"),
                "trust_remote_code": self.config.get("trust_remote_code", False)
            }
        
        print(f"🔧 Loading Vistral model: {model_id}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=pretrained_kwargs.get("trust_remote_code", False)
            )
            
            # Unified interface for pipeline
            self.processor = self.tokenizer
            
            # Configure pad_token if missing (Important for batch generation)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                print("  ⚙️  Configured pad_token_id = eos_token_id")
            
            # Set padding side to left for decoder-only models
            self.tokenizer.padding_side = "left"
            
            # Load model
            self.model = self.model_class.from_pretrained(
                model_id,
                **pretrained_kwargs
            ).eval()
            
            self.is_loaded = True
            print("✅ Vistral model and tokenizer loaded successfully!")
            
            if hasattr(self.model, 'hf_device_map'):
                print(f"📍 Device map: {self.model.hf_device_map}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Vistral model: {str(e)}")

    def _build_chat_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        """Return chat messages in Vistral format."""
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
            # Apply chat template (Step 1: Format)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize (Step 2: Encode)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            input_len = model_inputs["input_ids"].shape[-1]
            
            # Get generation kwargs
            if hasattr(self.config, 'get_generation_kwargs'):
                gen_kwargs = self.config.get_generation_kwargs()
            else:
                gen_kwargs = {
                    "max_new_tokens": self.config.get("max_tokens", 768),
                    "do_sample": self.config.get("do_sample", True),
                    "pad_token_id": self.tokenizer.pad_token_id,
                    # Vistral specific defaults from HF snippet if not in config
                    "top_p": self.config.get("top_p", 0.95),
                    "top_k": self.config.get("top_k", 40),
                    "repetition_penalty": self.config.get("repetition_penalty", 1.05)
                }
                if self.config.get("do_sample"):
                    gen_kwargs["temperature"] = self.config.get("temperature", 0.1)
            
            # Generate
            start_time = time.time()
            with torch.inference_mode(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
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
            if hasattr(self.config, 'get_generation_kwargs'):
                gen_kwargs = self.config.get_generation_kwargs()
            else:
                gen_kwargs = {
                    "max_new_tokens": self.config.get("max_tokens", 768),
                    "do_sample": self.config.get("do_sample", True),
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "top_p": self.config.get("top_p", 0.95),
                    "top_k": self.config.get("top_k", 40),
                    "repetition_penalty": self.config.get("repetition_penalty", 1.05)
                }
                if self.config.get("do_sample"):
                    gen_kwargs["temperature"] = self.config.get("temperature", 0.1)
            
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
                        add_generation_prompt=True
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
            with torch.inference_mode(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
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
        Extract JSON from Vistral's response.
        """
        # Step 1: Clean markdown code blocks
        response = raw_response.replace('', '').replace('```', '').strip()
        
        # Step 2: Find valid JSON object using brace matching
        first_brace = response.find('{')
        if first_brace != -1:
            open_braces = 0
            for i in range(first_brace, len(response)):
                if response[i] == '{':
                    open_braces += 1
                elif response[i] == '}':
                    open_braces -= 1
                    if open_braces == 0:
                        json_candidate = response[first_brace:i+1]
                        try:
                            json.loads(json_candidate)
                            return json_candidate.strip()
                        except json.JSONDecodeError:
                            continue
                            
        # Step 3: Try naive regex if brace matching failed
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            try:
                candidate = match.group(0)
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
                
        return response