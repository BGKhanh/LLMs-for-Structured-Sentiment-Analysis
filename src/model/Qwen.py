# src/model/Qwen.py

import re
import json
import time
import torch
from typing import Tuple, Dict, Any, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from .BaseModel import BaseModel


class QwenModel(BaseModel):
    """
    Qwen model implementation.
    
    Handles:
    - Qwen-specific loading (AutoModelForCausalLM)
    - Tokenizer configuration (pad_token setup)
    - Chat template application (simple message format)
    - Response extraction (handles thinking tokens: <think>...</think>)
    - Batch inference optimization
    
    Supports both thinking and non-thinking modes:
    - Thinking models: Qwen3-4B-Thinking-2507 (built-in thinking)
    - Non-thinking models: Qwen3-4B (enable_thinking parameter)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Qwen model.
        
        Args:
            config: Configuration dict with Qwen-specific settings
        
        Example config:
            {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "dtype": "auto",
                "device_map": "auto",
                "max_tokens": 512,
                "do_sample": False,
                "temperature": 0.1,
                "trust_remote_code": True,
                "enable_thinking": False  # Optional: for non-thinking models
            }
        """
        super().__init__(config)
        self.model_class = AutoModelForCausalLM
        self.tokenizer = None  # Qwen uses separate tokenizer
        self.chat_template_builder = self._build_chat_messages

    def load_model(self) -> None:
        """Load Qwen model and tokenizer."""
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
        
        print(f"🔧 Loading Qwen model: {model_id}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=kwargs.get("trust_remote_code", True)
            )
            
            # Configure pad_token if missing (Qwen-specific)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                print("  ⚙️  Configured pad_token_id = eos_token_id")
            
            # Set padding side to left (Qwen preference)
            self.tokenizer.padding_side = "left"
            
            # Load model
            self.model = self.model_class.from_pretrained(
                model_id,
                **kwargs
            ).eval()
            
            self.is_loaded = True
            print("✅ Qwen model and tokenizer loaded successfully!")
            
            if hasattr(self.model, 'hf_device_map'):
                print(f"📍 Device map: {self.model.hf_device_map}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen model: {str(e)}")

    def _build_chat_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        """Return chat messages in Qwen’s flat format."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _get_autocast_dtype(self) -> Union[torch.dtype, None]:
        """Helper to get torch dtype from config for autocast."""
        dtype_str = None
        
        # Extract dtype string from config
        if isinstance(self.config, dict):
            init_args = self.config.get("init_args", {})
            if isinstance(init_args, dict):
                dtype_str = init_args.get("dtype")
            else:
                dtype_str = getattr(init_args, "dtype", None)
        else:
            dtype_str = getattr(self.config.init_args, "dtype", None)
            
        # Convert string to torch.dtype
        if isinstance(dtype_str, str) and dtype_str != "auto" and hasattr(torch, dtype_str):
            return getattr(torch, dtype_str)
        return None  
            
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
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking  # For non-thinking models
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
                    "do_sample": self.config.get("do_sample", False),
                    "pad_token_id": self.tokenizer.pad_token_id
                }
                if self.config.get("do_sample"):
                    gen_kwargs["temperature"] = self.config.get("temperature", 0.1)
                    
            autocast_dtype = self._get_autocast_dtype()
            
            # Generate
            start_time = time.time()
            with torch.inference_mode(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=autocast_dtype):
                generated_ids = self.model.generate(
                    **model_inputs,
                    **gen_kwargs
                )
            
            generation_time = time.time() - start_time
            
            # Extract output tokens only
            output_ids = generated_ids[0][input_len:].tolist()
            
            # Decode (includes thinking tokens if present)
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            
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
                # Apply chat template to each message list
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

            autocast_dtype = self._get_autocast_dtype()
            
            # ===== BATCH GENERATION =====
            with torch.inference_mode(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=autocast_dtype):
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
                
            # Decode responses (includes thinking tokens if present)
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
        Extract JSON from Qwen's raw response.
        
        Qwen-specific handling:
        1. Remove thinking tokens (<think>...</think> blocks)
        2. Handle <|thought|> markers
        3. Extract from ```json ... ``` blocks
        4. Find first valid JSON object
        5. Fallback to cleaned response
        
        Args:
            raw_response: Raw text from Qwen model (may include thinking tokens)
        
        Returns:
            Extracted JSON string (thinking tokens removed)
        
        Note:
            Thinking content is automatically included in raw_response and will be
            logged/saved by debug_info system. This method extracts only the final
            JSON answer.
        """
        # Step 1: Remove <think>...</think> blocks (thinking tokens)
        think_block_pattern = r"<think>[\s\S]*?</think>\s*"
        response = re.sub(think_block_pattern, "", raw_response, flags=re.DOTALL)
        
        # Step 2: Handle <|thought|> marker (alternative thinking format)
        if "<|thought|>" in response:
            parts = response.split("<|thought|>")
            response = parts[-1].strip()  # Take content after last marker
        
        # Step 3: Try to extract from ```json ... ``` block
        match = re.search(r'```json\s*([\s\S]*?)\s*```', response, re.DOTALL)
        if match:
            json_candidate = match.group(1).strip()
            try:
                json.loads(json_candidate)  # Validate
                return json_candidate
            except json.JSONDecodeError:
                pass
        
        # Step 4: Find first valid JSON object using brace matching
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
                            json.loads(json_candidate)  # Validate
                            return json_candidate.strip()
                        except json.JSONDecodeError:
                            break
        
        # Step 5: Fallback - remove markdown and return
        return response.replace('```json', '').replace('```', '').strip()


# # Example usage (for reference):
# if __name__ == "__main__":
#     # Configuration
#     config = {
#         "model_id": "Qwen/Qwen2.5-7B-Instruct",
#         "dtype": "auto",
#         "device_map": "auto",
#         "max_tokens": 512,
#         "do_sample": False,
#         "trust_remote_code": True,
#         "enable_thinking": False  # Set True for non-thinking models with thinking mode
#     }
    
#     # Initialize and load
#     qwen = QwenModel(config)
#     qwen.load_model()
    
#     # Single inference
#     response, time_taken = qwen.generate_single(
#         system_prompt="You are a helpful assistant.",
#         user_prompt="Analyze this text..."
#     )
    
#     # Extract JSON (thinking tokens automatically removed)
#     json_str = qwen.extract_response(response)
    
#     # Cleanup
#     qwen.cleanup()