# src/model/SeaLLM.py

import re
import json
import time
from typing import Tuple, Dict, Any, List, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .BaseModel import BaseModel


class SeaLLModel(BaseModel):
    """SeaLLM model implementation using AutoModelForCausalLM/AutoTokenizer.

    - Flat chat template: [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    - 2-step tokenization: apply_chat_template(tokenize=False) -> tokenizer(...)
    - Batch generation with input/output token counting
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize SeaLLM model.

        Args:
            config: Model configuration (supports both dict and ModelConfig).
        """
        super().__init__(config)
        self.model_class = AutoModelForCausalLM
        self.tokenizer = None
        self.chat_template_builder = self._build_chat_messages

    def load_model(self) -> None:
        """Load SeaLLM model and tokenizer."""
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

        print(f"🔧 Loading SeaLLM model: {model_id}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=kwargs.get("trust_remote_code", True),
            )

            # Safe default: if pad_token missing, set to eos
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                print("  ⚙️  Configured pad_token_id = eos_token_id")

            self.tokenizer.padding_side = "left"
            
            # Load model
            self.model = self.model_class.from_pretrained(
                model_id,
                **kwargs,
            ).eval()

            self.is_loaded = True
            print("✅ SeaLLM model and tokenizer loaded successfully!")

            if hasattr(self.model, "hf_device_map"):
                print(f"📍 Device map: {self.model.hf_device_map}")

        except Exception as e:
            raise RuntimeError(f"Failed to load SeaLLM model: {str(e)}")

    def _build_chat_messages(
        self, system_prompt: str, user_prompt: str
    ) -> List[Dict[str, Any]]:
        """Return chat messages in flat format."""
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
        """Generate one response."""
        if not self.is_loaded:
            self.load_model()

        messages = self.chat_template_builder(system_prompt, user_prompt)

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,  # ignored if unsupported
            )

            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]

            # Generation kwargs
            if hasattr(self.config, "get_generation_kwargs"):
                gen_kwargs = self.config.get_generation_kwargs()
            else:
                gen_kwargs = {
                    "max_new_tokens": self.config.get("max_tokens", 512),
                    "do_sample": self.config.get("do_sample", False),
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                if self.config.get("do_sample"):
                    gen_kwargs["temperature"] = self.config.get("temperature", 0.1)
                    
            autocast_dtype = self._get_autocast_dtype()

            start_time = time.time()
            with torch.inference_mode(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=autocast_dtype):
                generated = self.model.generate(**inputs, **gen_kwargs)
            generation_time = time.time() - start_time

            output_ids = generated[0][input_len:].tolist()
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            return response, generation_time

        except Exception as e:
            print(f"❌ Error in generate_single: {str(e)}")
            return "{}", 0.0

    def generate_batch(
        self, inputs: Union[List[List[Dict[str, Any]]], Dict[str, torch.Tensor]]
    ) -> Tuple[List[str], float, List[int]]:
        """Batch generation with two input modes: tokenized dict or raw messages."""
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

            # Mode detection
            if isinstance(inputs, dict):
                tokenized_inputs = {
                    k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                    if k in ["input_ids", "attention_mask"]
                }
                input_length = tokenized_inputs["input_ids"].shape[1]
            else:
                texts = []
                for msgs in inputs:
                    t = self.tokenizer.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                    texts.append(t)

                tokenized_inputs = self.tokenizer(
                    texts, return_tensors="pt", padding=True
                ).to(self.model.device)
                input_length = tokenized_inputs["input_ids"].shape[1]
                
            autocast_dtype = self._get_autocast_dtype()

            with torch.inference_mode(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=autocast_dtype):
                generated_outputs = self.model.generate(
                    **tokenized_inputs, **gen_kwargs
                )

            batch_time = time.time() - start_time

            # Extract only new tokens
            output_ids = generated_outputs[:, input_length:]

            # Count output tokens per sample (stop at first PAD/EOS)
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
            output_token_counts: List[int] = []
            for i in range(output_ids.size(0)):
                seq = output_ids[i]
                count = 0
                for token_id in seq:
                    tid = token_id.item()
                    if (pad_token_id is not None and tid == pad_token_id) or (
                        eos_token_id is not None and tid == eos_token_id
                    ):
                        break
                    count += 1
                output_token_counts.append(count)

            batch_responses = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            return batch_responses, batch_time, output_token_counts

        except Exception as e:
            print(f"❌ Error in generate_batch: {str(e)}")
            if isinstance(inputs, dict):
                batch_size = inputs["input_ids"].shape[0]
            else:
                batch_size = len(inputs)
            return ["{}"] * batch_size, 0.0, [0] * batch_size

    def extract_response(self, raw_response: str) -> str:
        """Extract JSON from raw response (code fence or brace matching)."""
        # Try ```json ... ``` first
        match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_response, re.DOTALL)
        if match:
            json_candidate = match.group(1).strip()
            try:
                json.loads(json_candidate)
                return json_candidate
            except json.JSONDecodeError:
                pass

        # Fallback: brace matching for first valid JSON object
        first_brace = raw_response.find("{")
        if first_brace != -1:
            open_braces = 0
            for i in range(first_brace, len(raw_response)):
                if raw_response[i] == "{":
                    open_braces += 1
                elif raw_response[i] == "}":
                    open_braces -= 1
                    if open_braces == 0:
                        candidate = raw_response[first_brace : i + 1]
                        try:
                            json.loads(candidate)
                            return candidate.strip()
                        except json.JSONDecodeError:
                            break

        # Last resort: strip code fences
        return raw_response.replace("```json", "").replace("```", "").strip()