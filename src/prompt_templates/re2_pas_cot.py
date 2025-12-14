# src/prompt_templates/re2_pas_cot.py

from typing import Tuple, List, Dict, Any
import json
from .base import BasePromptTemplate

class Re2PaSCoTPrompt(BasePromptTemplate):
    """
    Combination of Re-reading (Re2) + Plan-and-Solve (PaS) + Chain-of-Thought (CoT).
    Standalone implementation.
    """
    
    # Copy & Adapt examples pool here (PaS style reasoning)
    EXAMPLES_POOL_VI = [ ... ] 
    EXAMPLES_POOL_EN = [ ... ]

    def __init__(self, eng: bool = False, n_shot: int = 0):
        super().__init__(eng)
        self.n_shot = n_shot
        self._selected_examples = None

    def prepare(self):
        # Logic select examples
        pass

    def _build_user_prompt(self, text, sent_id, **kwargs):
        # Logic combine 5 parts
        pass
        
    # ... other methods ...