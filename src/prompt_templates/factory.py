"""Registry-based factory for prompt creators."""

from __future__ import annotations

import inspect
from typing import Optional

from .contents import (
    FewShotCoTContent,
    FewShotContent,
    PlanAndSolveContent,
    ReReadingContent,
)
from .creator import PromptCreator

_REGISTRY: dict[str, type] = {
    "few_shot": FewShotContent,
    "few_shot_cot": FewShotCoTContent,
    "re_reading": ReReadingContent,
    "plan_and_solve": PlanAndSolveContent
}


def register_technique(name: str, content_cls: type) -> None:
    """Register a new content provider class for a technique name."""
    _REGISTRY[name] = content_cls


def build_prompt_creator(
    technique: str,
    language: str = "vi",
    n_shot: int = 0,
    plus_mode: bool = False,
    add_method: str = "none",
    examples_pool_path: Optional[str] = None,
) -> PromptCreator:
    """Build and prepare a PromptCreator from technique-level config."""
    if technique not in _REGISTRY:
        raise ValueError(
            f"Unknown technique: '{technique}'. Available: {sorted(_REGISTRY.keys())}"
        )

    content_cls = _REGISTRY[technique]
    sig = inspect.signature(content_cls.__init__)
    params = set(sig.parameters.keys()) - {"self"}

    kwargs: dict = {"language": language}
    if "n_shot" in params:
        kwargs["n_shot"] = n_shot
    if "plus" in params:
        kwargs["plus"] = plus_mode
    if "add_method" in params:
        kwargs["add_method"] = add_method

    content = content_cls(**kwargs)
    creator = PromptCreator(content=content, examples_pool_path=examples_pool_path)
    creator.prepare()
    return creator

