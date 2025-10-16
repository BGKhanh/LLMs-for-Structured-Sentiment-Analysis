# src/prompt_templates/base.py

from typing import Tuple, Optional
from abc import ABC, abstractmethod

class BasePromptTemplate(ABC):
    """
    Base class for all prompt templates.
    
    Design principles:
    - Lazy loading: Load resources only when needed
    - Caching: Cache invariant prompts (system_prompt)
    - Self-contained: No dependency on Dataset objects
    """
    
    def __init__(self, eng: bool = False):
        self.eng = eng
        self._system_prompt_cache = None  # Cache system prompt
        self._is_prepared = False
    
    @abstractmethod
    def _build_system_prompt(self) -> str:
        """Build system prompt (called once, then cached)."""
        pass
    
    @abstractmethod
    def _build_user_prompt(self, text: str, sent_id: str, **kwargs) -> str:
        """Build user prompt for each sample."""
        pass
    
    def prepare(self) -> None:
        """
        Prepare prompt template (load resources, build system prompt).
        
        This method should be called once before using get_prompt().
        Optional: get_prompt() will call it automatically if not prepared.
        """
        if self._is_prepared:
            return
        
        # Build and cache system prompt
        self._system_prompt_cache = self._build_system_prompt()
        self._is_prepared = True
        print(f"✅ {self.__class__.__name__} prepared")
    
    def get_prompt(
        self, 
        text: str, 
        sent_id: str,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Get (system_prompt, user_prompt) for a sample.
        
        Args:
            text: Text to analyze
            sent_id: Sample identifier
            **kwargs: Additional parameters for specific prompt types
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Lazy preparation
        if not self._is_prepared:
            self.prepare()
        
        # Build user prompt (varies per sample)
        user_prompt = self._build_user_prompt(text, sent_id, **kwargs)
        
        return self._system_prompt_cache, user_prompt