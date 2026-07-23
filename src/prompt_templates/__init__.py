"""
Public API for prompt templates.

This package currently exposes **two layers**:
- **New refactor API** (strategy-based): always available, does not depend on torch.
- **Legacy API** (class-per-technique): imported optionally for backward compatibility.

Do NOT delete legacy modules yet; keep them until the migration is verified.
"""

# ---------------------------------------------------------------------------
# New refactor API (always available)
# ---------------------------------------------------------------------------
from .shared import get_system_prompt, load_examples_pool
