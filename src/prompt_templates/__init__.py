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
from .contents import (
    ContentProvider,
    FewShotContent,
    FewShotCoTContent,
    ReReadingContent,
    PlanAndSolveContent,
)
from .creator import PromptCreator
from .factory import build_prompt_creator, register_technique
from .shared import get_system_prompt, load_examples_pool

__all__ = [
    # new API
    "ContentProvider",
    "FewShotContent",
    "FewShotCoTContent",
    "ReReadingContent",
    "PlanAndSolveContent",
    "PromptCreator",
    "build_prompt_creator",
    "register_technique",
    "get_system_prompt",
    "load_examples_pool",
]

# ---------------------------------------------------------------------------
# Legacy API (optional)
# ---------------------------------------------------------------------------
try:
    from .few_shot import FewShotPrompt
    from .zero_shot_CoT import ZeroShotCoTPrompt
    from .few_shot_CoT import FewShotCoTPrompt
    from .re_reading import ReReadingPrompt
    from .plan_and_solve import PlanAndSolvePrompt
    from .generators import SingleStagePromptGen, CoTStage1PromptGen, CoTStage2PromptGen
    from .re2_pas_cot import Re2PaSCoTPrompt

    __all__ += [
        "FewShotPrompt",
        "ZeroShotCoTPrompt",
        "FewShotCoTPrompt",
        "ReReadingPrompt",
        "PlanAndSolvePrompt",
        "SingleStagePromptGen",
        "CoTStage1PromptGen",
        "CoTStage2PromptGen",
        "Re2PaSCoTPrompt",
    ]
except ImportError:
    # Keep new API usable even when legacy dependencies (e.g., torch) are missing.
    pass