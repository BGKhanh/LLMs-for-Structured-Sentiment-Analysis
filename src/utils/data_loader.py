# src/utils/data_loader.py
import json
from typing import Dict, Any, List, Optional, Callable, Tuple
from torch.utils.data import Dataset
import torch
import numpy as np
import random


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for structured sentiment analysis.
    
    Designed for efficient batch processing with transformers and accelerate.
    Supports multiple prompt techniques 
    """
    
    def __init__(
        self,
        data_path: Optional[str],
        prompt_generator: Callable[[str, str], Tuple[str, str]],
        preloaded_data: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSON dataset file
            prompt_generator: Callable that takes (text, sent_id) and returns 
                             (system_prompt, user_prompt).
                             
                             Examples:
                             - FewShot: few_shot.get_prompt
                             - ZeroShot CoT Stage 1: lambda t, s: cot.get_prompt(t, s, stage="stage_1")
                             - ZeroShot CoT Stage 2: lambda t, s: cot.get_prompt(t, s, stage="stage_2", reasoning=reasoning_map[s])
            preloaded_data: List of samples already loaded, usually used for two-stage inference to avoid loading the same dataset twice

        Raises:
            FileNotFoundError: If data_path doesn't exist
            
        Note:
            The prompt_generator is responsible for:
            - Caching system prompts (if applicable)
            - Loading examples (if few-shot)
            - Handling stages (if CoT)
            - Managing reasoning (if CoT stage 2)
        """
        self.prompt_generator = prompt_generator
        if preloaded_data is not None:
            self.data = preloaded_data
            print(f"✅ Using preloaded dataset: {len(self.data)} samples")
        else:
            if not data_path:
                raise ValueError("Either data_path or preloaded_data must be provided.")
            self.data = self._load_data(data_path)
            
    def _load_data(self, data_path: str, preloaded_data: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Load dataset from JSON file."""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded {len(data)} samples from {data_path}")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {e}")
        
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample with prompts.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict containing:
                - text: Original text
                - sent_id: Sentence ID
                - system_prompt: System prompt for this sample
                - user_prompt: User prompt for this sample
                - opinions: Ground truth opinions (for evaluation)
        """
        sample = self.data[idx]
        text = sample.get("text", "")
        sent_id = str(sample.get("sent_id", f"unknown_{idx}"))
        
        # Simple interface - prompt_generator handles everything
        system_prompt, user_prompt = self.prompt_generator(text, sent_id)
        
        return {
            "text": text,
            "sent_id": sent_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "opinions": sample.get("opinions", [])  # For evaluation
        }


class SentimentCollator:
    """
    Custom collator for batching sentiment analysis samples.
    
    Applies chat template to batches for efficient processing.
    Compatible with transformers and accelerate.
    """
    
    def __init__(
            self,
            tokenizer,
            chat_template_builder: Callable[[str, str], List[Dict[str, Any]]],
            add_generation_prompt: bool = True,
            enable_thinking: bool = False,
        ):        
        """
        Initialize collator.
        
        Args:
            tokenizer: HuggingFace tokenizer/tokenizer
            chat_template_builder: Callable do model cung cấp để dựng message format
            add_generation_prompt: Có thêm generation prompt hay không
            enable_thinking: Bật chế độ thinking (tự động bị framework bỏ qua nếu model không hỗ trợ)
        """
        self.tokenizer = tokenizer
        self.add_generation_prompt = add_generation_prompt
        self.chat_template_builder = chat_template_builder
        self.enable_thinking = enable_thinking
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch of samples.
        
        Args:
            batch: List of samples from SentimentDataset
            
        Returns:
            Dict containing:
                - input_ids, attention_mask: Tokenized inputs ready for model
                - texts: Original texts
                - sent_ids: Sentence IDs
                - opinions: Ground truth opinions
                - system_prompts: System prompts for each sample  
                - user_prompts: User prompts for each sample      
        """
        # Extract components
        texts = [item["text"] for item in batch]
        sent_ids = [item["sent_id"] for item in batch]
        opinions = [item["opinions"] for item in batch]
        
        system_prompts = [item["system_prompt"] for item in batch]
        user_prompts = [item["user_prompt"] for item in batch]
        
        # Build messages for chat template
        batch_messages = [
            self.chat_template_builder(sys_prompt, usr_prompt)
            for sys_prompt, usr_prompt in zip(system_prompts, user_prompts)
        ]
        
        # Apply chat template with batching
        inputs = self.tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=self.add_generation_prompt,
            tokenize=False,
            enable_thinking=self.enable_thinking
        )
        if isinstance(inputs, list):
            tokenized = self.tokenizer(
                inputs,
                return_tensors="pt",
                padding=True
            )
        else:   
            tokenized = self.tokenizer(
                [inputs],
                return_tensors="pt",
                padding=True
            )

        return {
            **tokenized,  # input_ids, attention_mask, etc.
            "texts": texts,
            "sent_ids": sent_ids,
            "opinions": opinions,
            "system_prompts": system_prompts,
            "user_prompts": user_prompts       
        }

def seed_worker(worker_id):
    """
    Worker init function to ensure reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Helper function for easy DataLoader creation
def create_sentiment_dataloader(
    data_path: Optional[str],
    tokenizer,
    prompt_generator: Callable[[str, str], Tuple[str, str]],
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = False,
    preloaded_data: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = True,
    chat_template_builder: Optional[Callable[[str, str], List[Dict[str, Any]]]] = None,
    enable_thinking: bool = False,
    seed: int = 42,
):
    """
    Create DataLoader for sentiment analysis.
    
    Args:
        data_path: Path to dataset JSON
        tokenizer: HuggingFace tokenizer/tokenizer
        prompt_generator: Prompt generation callable
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        shuffle: Whether to shuffle data
        preloaded_data: List of samples already loaded
    Returns:
        DataLoader ready for inference
        
    Examples:
        >>> # Few-shot
        >>> from src.prompt_templates.few_shot import FewShotPrompt
        >>> few_shot = FewShotPrompt(eng=False, n_shot=3, examples_pool_path="data/train.json")
        >>> few_shot.prepare()
        >>> dataloader = create_sentiment_dataloader(
        ...     "data/test.json",
        ...     tokenizer,
        ...     few_shot.get_prompt,
        ...     batch_size=8
        ... )
        
        >>> # Zero-shot CoT Stage 1
        >>> from src.prompt_templates.zero_shot_CoT import ZeroShotCoTPrompt
        >>> cot = ZeroShotCoTPrompt(eng=False)
        >>> cot.prepare()
        >>> dataloader = create_sentiment_dataloader(
        ...     "data/test.json",
        ...     tokenizer,
        ...     lambda t, s: cot.get_prompt(t, s, stage="stage_1"),
        ...     batch_size=8
        ... )
        
        >>> # Zero-shot CoT Stage 2 (with reasoning)
        >>> reasoning_map = {...}  # From stage 1 inference
        >>> dataloader = create_sentiment_dataloader(
        ...     "data/test.json",
        ...     tokenizer,
        ...     lambda t, s: cot.get_prompt(t, s, stage="stage_2", reasoning=reasoning_map[s]),
        ...     batch_size=8
        ... )
    """
    from torch.utils.data import DataLoader
    
    dataset = SentimentDataset(
        data_path=data_path,
        prompt_generator=prompt_generator,
        preloaded_data=preloaded_data
    )
    
    if chat_template_builder is None:
        raise ValueError("chat_template_builder must be provided by the model.")
    
    collator = SentimentCollator(
        tokenizer=tokenizer,
        chat_template_builder=chat_template_builder,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking
    )
    
    # Create generator for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,  # For faster GPU transfer
        worker_init_fn=seed_worker,
        generator=g
    )
    
    return dataloader