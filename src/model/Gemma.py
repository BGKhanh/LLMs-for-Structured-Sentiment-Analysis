# src/model/gemma.py

import re
import json
import time
import torch
from typing import Tuple, Dict, Any, List
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
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
                "torch_dtype": torch.float32,
                "device_map": "auto",
                "max_tokens": 2048,
                "do_sample": False,
                "temperature": 0.1,
                "trust_remote_code": True
            }
        """
        super().__init__(config)
        self.model_class = Gemma3ForConditionalGeneration
    
    def load_model(self) -> None:
        """Load Gemma 3 model and processor."""
        if self.is_loaded:
            print("✅ Model already loaded!")
            return
        
        model_id = self.config["model_id"]
        print(f"🔧 Loading Gemma model: {model_id}")
        
        try:
            # Load model
            self.model = self.model_class.from_pretrained(
                model_id,
                device_map=self.config.get("device_map", "auto"),
                torch_dtype=self.config.get("torch_dtype", torch.float32),
                trust_remote_code=self.config.get("trust_remote_code", True)
            ).eval()
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            self.is_loaded = True
            print("✅ Gemma model and processor loaded successfully!")
            
            # Print device info
            if hasattr(self.model, 'hf_device_map'):
                print(f"📍 Device map: {self.model.hf_device_map}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Gemma model: {str(e)}")
    
    def generate_single(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Tuple[str, float]:
        """
        Generate response for single input.
        
        Args:
            system_prompt: System instruction
            user_prompt: User query
        
        Returns:
            Tuple of (raw_response, generation_time_seconds)
        """
        if not self.is_loaded:
            self.load_model()
        
        # Build messages in Gemma chat format
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]
        
        try:
            # Apply chat template and tokenize
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Generate
            start_time = time.time()
            with torch.inference_mode(), torch.autocast(device_type="cuda"):
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get("max_tokens", 2048),
                    do_sample=self.config.get("do_sample", False),
                    temperature=self.config.get("temperature", 0.1) if self.config.get("do_sample") else None
                )
                # Extract only the generated part (exclude input)
                generation = generation[0][input_len:]
            
            generation_time = time.time() - start_time
            
            # Decode
            response = self.processor.decode(generation, skip_special_tokens=True)
            
            return response, generation_time
            
        except Exception as e:
            print(f"❌ Error in generate_single: {str(e)}")
            return "{}", 0.0
    
    def generate_batch(
        self,
        batch_messages: List[List[Dict[str, Any]]]
    ) -> Tuple[List[str], float]:
        """
        Generate responses for batch inputs (optimized for throughput).
        
        Args:
            batch_messages: List of message lists
                Example: [
                    [
                        {"role": "system", "content": [{"type": "text", "text": "..."}]},
                        {"role": "user", "content": [{"type": "text", "text": "..."}]}
                    ],
                    ...
                ]
        
        Returns:
            Tuple of (list_of_raw_responses, total_batch_time)
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Apply chat template for batch
            start_time = time.time()
            inputs = self.processor.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                return_dict=True
            ).to(self.model.device)
            
            # Batch generation
            with torch.inference_mode(), torch.autocast(device_type="cuda"):
                generated_outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.get("max_tokens", 2048),
                    do_sample=self.config.get("do_sample", False),
                    temperature=self.config.get("temperature", 0.1) if self.config.get("do_sample") else None
                )
            
            batch_time = time.time() - start_time
            
            # Extract generated tokens (exclude input)
            output_ids = generated_outputs[:, inputs.input_ids.shape[1]:]
            
            # Decode batch
            batch_responses = self.processor.batch_decode(output_ids, skip_special_tokens=True)
            
            return batch_responses, batch_time
            
        except Exception as e:
            print(f"❌ Error in generate_batch: {str(e)}")
            # Return empty responses for all items in batch
            return ["{}"] * len(batch_messages), 0.0
    
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
#         "torch_dtype": torch.float32,
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