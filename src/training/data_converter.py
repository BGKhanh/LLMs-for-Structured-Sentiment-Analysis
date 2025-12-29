# src/training/data_converter.py

"""
Training data conversion utilities.

Reuses SentimentDataset for data loading and prompt generation,
then converts to SFTTrainer format.
"""

from src.utils.data_loader import SentimentDataset
from torch.utils.data import Dataset
import json
from typing import Callable


class TrainingSentimentDataset(Dataset):
    """
    Training dataset wrapper for structured sentiment analysis.
    
    Reuses SentimentDataset infrastructure, adds training format conversion.
    """
    
    def __init__(
        self,
        data_path: str,
        prompt_generator: Callable,
        tokenizer
    ):
        # Reuse existing infrastructure!
        self.base_dataset = SentimentDataset(
            data_path=data_path,
            prompt_generator=prompt_generator
        )
        self.tokenizer = tokenizer
        print(f"✅ Initialized TrainingSentimentDataset: {len(self)} samples")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get from base (has prompts + opinions)
        sample = self.base_dataset[idx]
        
        # Build training format
        messages = [
            {"role": "system", "content": sample["system_prompt"]},
            {"role": "user", "content": sample["user_prompt"]},
            {"role": "assistant", "content": json.dumps(sample["opinions"], ensure_ascii=False)}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {"text": text}


def prepare_training_datasets(
    train_path: str,
    eval_path: str,
    prompt_generator: Callable,
    tokenizer
):
    """
    Prepare training and eval datasets.
    
    Args:
        train_path: Path to training data
        eval_path: Path to eval data (optional)
        prompt_generator: Prompt template's get_prompt
        tokenizer: Tokenizer
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    train_ds = TrainingSentimentDataset(train_path, prompt_generator, tokenizer)
    
    eval_ds = None
    if eval_path:
        eval_ds = TrainingSentimentDataset(eval_path, prompt_generator, tokenizer)
    
    return train_ds, eval_ds