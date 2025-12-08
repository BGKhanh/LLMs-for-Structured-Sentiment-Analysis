# src/model/base.py

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
import torch
import gc


class BaseModel(ABC):
    """
    Base class for all LLM models.
    
    Design principles:
    - Model-agnostic interface
    - Each model handles its own response format
    - Config-driven initialization
    - GPU memory management
    """
    
    def __init__(self, config: Any):
        """
        Initialize base model.
        
        Args:
            config: Model configuration dict containing:
                - model_id: HuggingFace model ID
                - dtype: torch.float32, torch.float16, torch.bfloat16
                - device_map: "auto", "cpu", or specific GPU mapping
                - max_tokens: Max new tokens to generate
                - do_sample: Whether to sample or use greedy
                - temperature: Sampling temperature (if do_sample=True)
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        if isinstance(self.config, dict):
            # Dict mode (legacy)
            gen_args = self.config.get("generation_args", {})
            if isinstance(gen_args, dict):
                self.enable_thinking = gen_args.get("enable_thinking", False)
            else:
                self.enable_thinking = getattr(gen_args, "enable_thinking", False)
        else:
            # Object mode (ModelConfig)
            self.enable_thinking = getattr(self.config.generation_args, "enable_thinking", False)
           
    @abstractmethod
    def load_model(self) -> None:
        """
        Load model and tokenizer.
        
        Must be implemented by each model class.
        """
        pass
    
    @abstractmethod
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
            Tuple of (raw_response, generation_time)
        
        Must be implemented by each model class.
        """
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        batch_messages: List[List[Dict[str, Any]]]
    ) -> Tuple[List[str], float, List[int]]:
        """
        Generate responses for batch inputs.
        
        Args:
            batch_messages: List of message lists (chat template format)
        
        Returns:
            Tuple of (list_of_responses, total_generation_time)
        
        Must be implemented by each model class.
        """
        pass
    
    @abstractmethod
    def extract_response(self, raw_response: str) -> str:
        """
        Extract JSON from model's raw response.
        
        Each model may have different response format, so this
        must be implemented specifically.
        
        Args:
            raw_response: Raw text from model
        
        Returns:
            Extracted JSON string
        
        Must be implemented by each model class.
        """
        pass
    
    def cleanup(self) -> None:
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            
            # Aggressive cleanup for multi-GPU
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        
        print("🧹 GPU memory cleanup completed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        # Update to reflect new config structure safely
        if isinstance(self.config, dict):
            model_id = self.config.get("init_args", {}).get("model_id", "unknown")
            dtype = self.config.get("init_args", {}).get("dtype", "unknown")
        else:
            model_id = self.config.init_args.model_id
            dtype = self.config.init_args.dtype
            
        return {
            "model_id": model_id,
            "is_loaded": self.is_loaded,
            "dtype": str(dtype),
        }