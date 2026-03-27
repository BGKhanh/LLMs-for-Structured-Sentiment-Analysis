"""PromptCreator: single mechanism owner for prompt preparation/caching."""

from __future__ import annotations

from typing import Optional

from .contents import ContentProvider
from .shared import get_system_prompt, load_examples_pool


class PromptCreator:
    """Create `(system_prompt, user_prompt)` via pluggable content provider.

    External compatibility:
    - supports `prepare()`
    - supports `get_prompt(text, sent_id)`
    - provides `_system_prompt_cache` alias used by lm-eval utils
    """

    def __init__(self, content: ContentProvider, examples_pool_path: Optional[str] = None):
        self._content = content
        self._pool_path = examples_pool_path
        self._system_prompt = get_system_prompt(content.language)
        self._prepared = False

    def prepare(self) -> None:
        """Load pool and initialize content provider exactly once."""
        if self._prepared:
            return
        pool = load_examples_pool(self._pool_path, self._content.language)
        self._content.setup(pool)
        self._prepared = True

    def get_prompt(self, text: str, sent_id: str):
        """Return `(system_prompt, user_prompt)` for one sample."""
        if not self._prepared:
            self.prepare()
        return self._system_prompt, self._content.user_prompt(text, sent_id)

    @property
    def _system_prompt_cache(self) -> str:
        """Backward-compatible alias used by existing lm-eval integration."""
        return self._system_prompt

