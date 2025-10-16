# src/utils/data_loader.py
import json
from typing import Dict, Any, List, Optional, Callable, Tuple
from torch.utils.data import Dataset
import torch


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for structured sentiment analysis.
    
    Designed for efficient batch processing with transformers and accelerate.
    Supports multiple prompt techniques 
    """
    
    def __init__(
        self,
        data_path: str,
        prompt_generator: Callable[[str, str], Tuple[str, str]]
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
        
        Raises:
            FileNotFoundError: If data_path doesn't exist
            
        Note:
            The prompt_generator is responsible for:
            - Caching system prompts (if applicable)
            - Loading examples (if few-shot)
            - Handling stages (if CoT)
            - Managing reasoning (if CoT stage 2)
        """
        self.data = self._load_data(data_path)
        self.prompt_generator = prompt_generator
    
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
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
    
    def __init__(self, processor, add_generation_prompt: bool = True):
        """
        Initialize collator.
        
        Args:
            processor: HuggingFace processor/tokenizer
            add_generation_prompt: Whether to add generation prompt
        """
        self.processor = processor
        self.add_generation_prompt = add_generation_prompt
    
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
        """
        # Extract components
        texts = [item["text"] for item in batch]
        sent_ids = [item["sent_id"] for item in batch]
        opinions = [item["opinions"] for item in batch]
        
        # Build messages for chat template
        batch_messages = []
        for item in batch:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": item["system_prompt"]}]},
                {"role": "user", "content": [{"type": "text", "text": item["user_prompt"]}]}
            ]
            batch_messages.append(messages)
        
        # Apply chat template with batching
        inputs = self.processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=self.add_generation_prompt,
            tokenize=True,
            return_tensors="pt",
            padding=True,
            return_dict=True
        )
        
        return {
            **inputs,  # input_ids, attention_mask, etc.
            "texts": texts,
            "sent_ids": sent_ids,
            "opinions": opinions
        }



# Helper function for easy DataLoader creation
def create_sentiment_dataloader(
    data_path: str,
    processor,
    prompt_generator: Callable[[str, str], Tuple[str, str]],
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = False
):
    """
    Create DataLoader for sentiment analysis.
    
    Args:
        data_path: Path to dataset JSON
        processor: HuggingFace processor/tokenizer
        prompt_generator: Prompt generation callable
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader ready for inference
        
    Examples:
        >>> # Few-shot
        >>> from src.prompt_templates.few_shot import FewShotPrompt
        >>> few_shot = FewShotPrompt(eng=False, n_shot=3, examples_pool_path="data/train.json")
        >>> few_shot.prepare()
        >>> dataloader = create_sentiment_dataloader(
        ...     "data/test.json",
        ...     processor,
        ...     few_shot.get_prompt,
        ...     batch_size=8
        ... )
        
        >>> # Zero-shot CoT Stage 1
        >>> from src.prompt_templates.zero_shot_CoT import ZeroShotCoTPrompt
        >>> cot = ZeroShotCoTPrompt(eng=False)
        >>> cot.prepare()
        >>> dataloader = create_sentiment_dataloader(
        ...     "data/test.json",
        ...     processor,
        ...     lambda t, s: cot.get_prompt(t, s, stage="stage_1"),
        ...     batch_size=8
        ... )
        
        >>> # Zero-shot CoT Stage 2 (with reasoning)
        >>> reasoning_map = {...}  # From stage 1 inference
        >>> dataloader = create_sentiment_dataloader(
        ...     "data/test.json",
        ...     processor,
        ...     lambda t, s: cot.get_prompt(t, s, stage="stage_2", reasoning=reasoning_map[s]),
        ...     batch_size=8
        ... )
    """
    from torch.utils.data import DataLoader
    
    dataset = SentimentDataset(
        data_path=data_path,
        prompt_generator=prompt_generator
    )
    
    collator = SentimentCollator(processor)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True  # For faster GPU transfer
    )
    
    return dataloader