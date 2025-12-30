# src/training/data_converter.py

"""
Training data conversion utilities.

Converts SemEval dataset to HuggingFace Dataset format for SFTTrainer.
"""

from src.utils.data_loader import SentimentDataset
from datasets import Dataset  # ✅ HuggingFace Dataset
import json
from typing import Callable, Tuple, Optional


def create_training_dataset(
    data_path: str,
    prompt_generator: Callable,
    tokenizer
) -> Dataset:
    """
    Create HuggingFace Dataset for SFT training.
    
    Args:
        data_path: Path to SemEval JSON dataset
        prompt_generator: Prompt template's get_prompt method
        tokenizer: Tokenizer for chat template
        
    Returns:
        HuggingFace Dataset with "text" column
    """
    # Load data using existing infrastructure
    sentiment_ds = SentimentDataset(
        data_path=data_path,
        prompt_generator=prompt_generator
    )
    
    # Convert to list of formatted conversations
    conversations = []
    for i in range(len(sentiment_ds)):
        sample = sentiment_ds[i]
        
        # Build messages
        messages = [
            {"role": "system", "content": sample["system_prompt"]},
            {"role": "user", "content": sample["user_prompt"]},
            {"role": "assistant", "content": json.dumps(sample["opinions"], ensure_ascii=False)}
        ]
        
        # Apply chat template
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            # Fallback: merge system into user if system role not supported
            print(f"⚠️  Chat template failed for sample {i}, using fallback")
            messages = [
                {
                    "role": "user",
                    "content": f"{sample['system_prompt']}\n\n{sample['user_prompt']}"
                },
                {
                    "role": "assistant",
                    "content": json.dumps(sample["opinions"], ensure_ascii=False)
                }
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        
        conversations.append({"text": text})
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(conversations)
    print(f"✅ Created HuggingFace Dataset: {len(dataset)} samples")
    
    return dataset


def prepare_training_datasets(
    train_path: str,
    eval_path: Optional[str],
    prompt_generator: Callable,
    tokenizer
) -> Tuple[Dataset, Optional[Dataset]]:
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
    print("\n📊 Preparing Training Datasets")
    print("="*70)
    
    train_dataset = create_training_dataset(train_path, prompt_generator, tokenizer)
    
    eval_dataset = None
    if eval_path:
        eval_dataset = create_training_dataset(eval_path, prompt_generator, tokenizer)
    
    return train_dataset, eval_dataset