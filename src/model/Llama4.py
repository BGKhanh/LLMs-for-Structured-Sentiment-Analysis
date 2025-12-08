# src/model/Llama4.py

import re
import json
import time
import torch
from typing import Tuple, Dict, Any, List, Union
from transformers import Llama4ForConditionalGeneration, AutoTokenizer
from .BaseModel import BaseModel


class Llama4Model(BaseModel):
    """
    Llama 4 model implementation.
    
    Handles:
    - Llama 4 specific loading (Llama4ForConditionalGeneration)
    - Nested chat template structure (similar to Gemma 3)
    - Batch processing with token tracking
    
    Reference: meta-llama/Llama-4-Scout-17B-16E-Instruct
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Llama 4 model.
        
        Args:
            config: Configuration dict with Llama-specific settings
        """
        super().__init__(config)
        self.model_class = Llama4ForConditionalGeneration
        self.tokenizer = None
        self.chat_template_builder = self._build_chat_messages

    def load_model(self) -> None:
        """Load Llama 4 model and tokenizer."""
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

        print(f"🔧 Loading Llama 4 model: {model_id}")
        
        try:
            # Load tokenizer (Replacing Processor as requested)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Configure pad_token if missing
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                print("  ⚙️  Configured pad_token_id = eos_token_id")
            
            self.tokenizer.padding_side = "left"
            
            # Load model
            self.model = self.model_class.from_pretrained(
                model_id,
                **kwargs
            ).eval()
            
            self.is_loaded = True
            print("✅ Llama 4 model and tokenizer loaded successfully!")
            
            if hasattr(self.model, 'hf_device_map'):
                print(f"📍 Device map: {self.model.hf_device_map}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Llama 4 model: {str(e)}")

    def _build_chat_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        """
        Return chat messages in Llama 4's nested content format.
        Similar to Gemma 3, it expects content to be a list of dicts.
        """
        return [
            {
                "role": "system", 
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user", 
                "content": [{"type": "text", "text": user_prompt}]
            }
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
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.enable_thinking
            )
            
            # Tokenize
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]
            
            # Get generation kwargs
            if hasattr(self.config, 'get_generation_kwargs'):
                gen_kwargs = self.config.get_generation_kwargs()
            else:
                gen_kwargs = {
                    "max_new_tokens": self.config.get("max_tokens", 256),
                    "do_sample": self.config.get("do_sample", True),
                }
                if self.config.get("do_sample"):
                    gen_kwargs["temperature"] = self.config.get("temperature", 0.6) # Default slightly higher for Llama
            
            # Generate
            start_time = time.time()
            with torch.inference_mode(), torch.autocast(device_type="cuda"):
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs  
                )
                # Slice output to get only new tokens
                generation = outputs[0][input_len:]
            
            generation_time = time.time() - start_time
            response = self.tokenizer.decode(generation, skip_special_tokens=True).strip()
            
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
                - List[List[Dict]]: Raw messages for chat template
        
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
                    
                tokenized_inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                ).to(self.model.device)
                input_length = tokenized_inputs['input_ids'].shape[1]
            
            # ===== BATCH GENERATION =====
            with torch.inference_mode(), torch.autocast(device_type="cuda"):
                generated_outputs = self.model.generate(
                    **tokenized_inputs,
                    **gen_kwargs
                )
            
            batch_time = time.time() - start_time
            
            # ===== DECODE OUTPUTS =====
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
            
            # Clean up responses
            batch_responses = [resp.strip() for resp in batch_responses]
            
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
        Extract JSON from Llama 4's raw response.
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
        
        # Step 2: Find first valid JSON object
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