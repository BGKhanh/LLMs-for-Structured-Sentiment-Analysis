# src/model/gemma.py

import re
import json
import time
import torch
from typing import Tuple, Dict, Any, List, Union
from transformers import Gemma3ForConditionalGeneration, AutoTokenizer  
from .BaseModel import BaseModel


class GemmaModel(BaseModel):
    """
    Gemma 3 model implementation.
    
    Handles:
    - Gemma-specific loading
    - Chat template application
    - Response extraction (handles ```json and assistant markers)
    - Batch inference optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Gemma model.
        
        Args:
            config: Configuration dict with Gemma-specific settings
        
        Example config:
            {
                "model_id": "google/gemma-3-4b-it",
                "dtype": torch.float32,
                "device_map": "auto",
                "max_tokens": 2048,
                "do_sample": False,
                "temperature": 0.1,
                "trust_remote_code": True
            }
        """
        super().__init__(config)
        self.model_class = Gemma3ForConditionalGeneration
        self.tokenizer = None
        self.chat_template_builder = self._build_chat_messages
        
    def load_model(self) -> None:
        """Load Gemma 3 model and tokenizer."""
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

        print(f"🔧 Loading Gemma model: {model_id}")
        
        
        try:
            # Load model
            self.model = self.model_class.from_pretrained(
                model_id,
                **kwargs
            ).eval()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            self.is_loaded = True
            print("✅ Gemma model and tokenizer loaded successfully!")
            
            if hasattr(self.model, 'hf_device_map'):
                print(f"📍 Device map: {self.model.hf_device_map}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Gemma model: {str(e)}")
        
    def _build_chat_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        """Return chat messages in Gemma’s nested content format."""
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]

    def generate_single(self, system_prompt: str, user_prompt: str) -> Tuple[str, float]:
        """Generate response for single input."""
        if not self.is_loaded:
            self.load_model()
        
        messages = self.chat_template_builder(system_prompt, user_prompt)
        
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.enable_thinking
            )
            
            inputs = self.tokenizer([text],
                return_tensors="pt"
            ).to(self.model.device)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Get generation kwargs
            if hasattr(self.config, 'get_generation_kwargs'):
                # ModelConfig object
                gen_kwargs = self.config.get_generation_kwargs()
            else:
                # Dict (backward compatible)
                gen_kwargs = {
                    "max_new_tokens": self.config.get("max_tokens", 2048),
                    "do_sample": self.config.get("do_sample", False),
                }
                if self.config.get("do_sample"):
                    gen_kwargs["temperature"] = self.config.get("temperature", 0.1)
            
            # Generate
            start_time = time.time()
            with torch.inference_mode(), torch.autocast(device_type="cuda"):
                generation = self.model.generate(
                    **inputs,
                    **gen_kwargs  
                )
                generation = generation[0][input_len:]
            
            generation_time = time.time() - start_time
            response = self.tokenizer.decode(generation, skip_special_tokens=True)
            
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
        
        Supports two input modes:
        1. Tokenized inputs (Dict) - from DataLoader (efficient, no re-tokenization)
        2. Raw messages (List) - for manual batching (e.g., multi-stage CoT)
        
        Args:
            inputs: Either:
                - Dict[str, torch.Tensor]: Tokenized inputs from collator
                  {'input_ids': tensor, 'attention_mask': tensor}
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
            
            print(f"🔧 gen_kwargs: {gen_kwargs}")
            start_time = time.time()
            
            # ===== MODE DETECTION =====
            if isinstance(inputs, dict):
                # Mode 1: Tokenized inputs from DataLoader
                # Already tokenized, just move to device
                tokenized_inputs = {
                    k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                    if k in ['input_ids', 'attention_mask']
                }
                input_length = tokenized_inputs['input_ids'].shape[1]
                
            else:
                # Mode 2: Raw messages (manual batching)
                # Need to apply chat template
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
                    texts,  # List of message lists
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
            # Extract only new tokens (remove input)
            output_ids = generated_outputs[:, input_length:]
            
            # Get special token IDs for proper counting
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
        
            # Count output tokens per sample (exclude PAD and EOS)
            output_token_counts = []
            for i in range(output_ids.size(0)):
                seq = output_ids[i]
                count = 0
                for token_id in seq:
                    tid = token_id.item()
                    # Stop counting at first PAD or EOS token
                    if (pad_token_id is not None and tid == pad_token_id) or \
                    (eos_token_id is not None and tid == eos_token_id):
                        break
                    count += 1
                output_token_counts.append(count)
                
            batch_responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
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
        Extract JSON from Gemma's raw response.
        
        Gemma-specific handling:
        1. Remove "assistant:" markers
        2. Extract from ```json ... ``` blocks
        3. Find first valid JSON object
        4. Fallback to cleaned response
        
        Args:
            raw_response: Raw text from Gemma model
        
        Returns:
            Extracted JSON string
        """
        # Step 1: Handle Gemma assistant markers
        assistant_markers = ["assistant:", "assistant", "<assistant>"]
        for marker in assistant_markers:
            if marker in raw_response:
                raw_response = raw_response.split(marker, 1)[1].strip()
        
        # Step 2: Try to extract from ```json ... ``` block
        match = re.search(r'```json\s*([\s\S]*?)\s*```', raw_response, re.DOTALL)
        if match:
            json_candidate = match.group(1).strip()
            try:
                json.loads(json_candidate)  # Validate
                return json_candidate
            except json.JSONDecodeError:
                pass
        
        # Step 3: Find first valid JSON object
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
                            json.loads(json_candidate)  # Validate
                            return json_candidate.strip()
                        except json.JSONDecodeError:
                            break
        
        # Step 4: Fallback - remove markdown and return
        return raw_response.replace('```json', '').replace('```', '').strip()


# # Example usage (for reference):
# if __name__ == "__main__":
#     # Configuration
#     config = {
#         "model_id": "google/gemma-3-4b-it",
#         "dtype": torch.float32,
#         "device_map": "auto",
#         "max_tokens": 2048,
#         "do_sample": False,
#         "trust_remote_code": True
#     }
    
#     # Initialize and load
#     gemma = GemmaModel(config)
#     gemma.load_model()
    
#     # Single inference
#     response, time_taken = gemma.generate_single(
#         system_prompt="You are a helpful assistant.",
#         user_prompt="Analyze this text..."
#     )
    
#     # Extract JSON
#     json_str = gemma.extract_response(response)
    
#     # Cleanup
#     gemma.cleanup()