"""Content providers for prompt construction strategy.

Each provider only handles *user prompt content*.
Caching/system prompt/loading pools are handled by PromptCreator.
"""

from __future__ import annotations

import random
from typing import Any, Protocol, runtime_checkable

from .blocks import (
    base_question_block,
    cot_demo_block,
    few_shot_block,
    few_shot_question_block,
    pas_instruction_block,
    rereading_block,
)


@runtime_checkable
class ContentProvider(Protocol):
    """Contract for prompt content providers."""

    language: str

    def setup(self, pool: list[dict[str, Any]]) -> None:
        """Prepare provider state from pre-loaded pool."""

    def user_prompt(self, text: str, sent_id: str) -> str:
        """Build user prompt for one sample."""


class FewShotContent:
    """Few-shot content using legacy FewShotPrompt formatting."""

    def __init__(self, language: str, n_shot: int = 0):
        self.language = language
        self.n_shot = max(0, int(n_shot))
        self._examples: list[dict[str, Any]] = []

    def setup(self, pool: list[dict[str, Any]]) -> None:
        if self.n_shot <= 0:
            return
        if len(pool) < self.n_shot:
            raise ValueError(
                f"Not enough examples. Requested {self.n_shot}, but only {len(pool)} available."
            )
        # Keep legacy behavior: random.sample fixed once during prepare/setup.
        self._examples = random.sample(pool, self.n_shot)

    def user_prompt(self, text: str, sent_id: str) -> str:
        if self._examples:
            return "\n\n".join(
                [
                    few_shot_block(self._examples, self.language),
                    few_shot_question_block(text, sent_id, self.language),
                ]
            )
        return base_question_block(text, sent_id, self.language)


class FewShotCoTContent:
    """Few-shot CoT with reasoning demos (legacy FewShotCoTPrompt style)."""

    def __init__(self, language: str, n_shot: int = 3):
        self.language = language
        self.n_shot = max(0, int(n_shot))
        self._examples: list[dict[str, Any]] = []

    def setup(self, pool: list[dict[str, Any]]) -> None:
        if self.n_shot <= 0:
            return
        if self.n_shot > len(pool):
            raise ValueError(
                f"Requested n_shot={self.n_shot} but only {len(pool)} examples available in pool"
            )
        # Keep legacy behavior: fixed first n examples.
        self._examples = pool[: self.n_shot]

    def user_prompt(self, text: str, sent_id: str) -> str:
        parts: list[str] = []
        if self._examples:
            parts.append(cot_demo_block(self._examples, self.language))
            parts.append(
                "\nNow, analyze the following case:\n"
                if self.language == "en"
                else "\nBây giờ, hãy phân tích trường hợp sau:\n"
            )
        if self.language == "en":
            parts.append(
                f"""Input: "{text}" (sent_id: {sent_id})

Think step by step, then return the final JSON."""
            )
        else:
            parts.append(f'Input: "{text}" (sent_id: {sent_id})')
        return "".join(parts)


class ReReadingContent:
    """Re-reading content with optional augmentation methods."""

    def __init__(self, language: str, add_method: str = "none", n_shot: int = 0):
        self.language = language
        self.add_method = add_method
        self.n_shot = max(0, int(n_shot))
        self._examples: list[dict[str, Any]] = []

    def setup(self, pool: list[dict[str, Any]]) -> None:
        if self.n_shot <= 0:
            return
        if self.add_method in ("FewShot", "FewShot_CoT", "few_shot", "cot_demo"):
            k = min(self.n_shot, len(pool))
            if self.add_method in ("FewShot_CoT", "cot_demo"):
                self._examples = pool[:k]
            else:
                self._examples = random.sample(pool, k)

    def _trigger(self) -> str:
        if self.language == "en":
            if self.add_method == "0_CoT":
                return "Let's think step by step."
            if self.add_method in ("PaS", "pas"):
                return (
                    "Let's first understand the problem and devise a plan to solve the problem. "
                    "Then, let's carry out the plan, solve the problem step by step, "
                    "and give the ultimate answer. Please explicitly generate the mentioned process: "
                    "[Problem Understanding], [Plan], [Detailed Analysis], [Answer]."
                )
            return ""
        if self.add_method == "0_CoT":
            return "Hãy cùng suy nghĩ từng bước."
        if self.add_method in ("PaS", "pas"):
            return (
                "Đầu tiên hãy hiểu vấn đề và vạch ra kế hoạch để giải quyết. "
                "Sau đó, hãy thực hiện kế hoạch, phân tích từng bước, "
                "và đưa ra câu trả lời cuối cùng. "
                "Vui lòng sinh ra quy trình cụ thể theo các bước: "
                "[Hiểu vấn đề], [Lập kế hoạch], [Phân tích chi tiết], [Câu trả lời]."
            )
        return ""

    def user_prompt(self, text: str, sent_id: str) -> str:
        parts: list[str] = []
        if self._examples:
            if self.add_method in ("FewShot_CoT", "cot_demo"):
                parts.append(cot_demo_block(self._examples, self.language))
            else:
                parts.append(few_shot_block(self._examples, self.language))
            parts.append(
                "\nNow, analyze the following case:\n"
                if self.language == "en"
                else "\nBây giờ, hãy phân tích trường hợp sau:\n"
            )

        core = rereading_block(text, sent_id, self.language)
        trigger = self._trigger()
        if trigger:
            core = f"{core}\n\n{trigger}"
        parts.append(core)
        return "".join(parts)


class PlanAndSolveContent:
    """Plan-and-Solve / Plan-and-Solve+ content."""

    def __init__(self, language: str, plus: bool = False, n_shot: int = 0):
        self.language = language
        self.plus = bool(plus)
        self.n_shot = max(0, int(n_shot))
        self._examples: list[dict[str, Any]] = []

    def setup(self, pool: list[dict[str, Any]]) -> None:
        # Keep legacy behavior: only load examples when plus and n_shot > 0.
        if self.plus and self.n_shot > 0:
            k = min(self.n_shot, len(pool))
            self._examples = pool[:k]

    def user_prompt(self, text: str, sent_id: str) -> str:
        parts: list[str] = [pas_instruction_block(plus=self.plus, language=self.language)]
        if self._examples:
            title = (
                "HERE ARE SOME DEMONSTRATION EXAMPLES (PLEASE FOLLOW SIMILAR PROCESS):"
                if self.language == "en"
                else "DƯỚI ĐÂY LÀ MỘT SỐ VÍ DỤ MINH HỌA (HÃY LÀM THEO QUY TRÌNH TƯƠNG TỰ):"
            )
            parts.append("\n\n" + title + "\n\n" + cot_demo_block(self._examples, self.language))
            parts.append(
                "\nNow, analyze the following case:\n"
                if self.language == "en"
                else "\nBây giờ, hãy phân tích trường hợp sau:\n"
            )

        if self.language == "en":
            parts.append(f'Analyze the sentiment for the following text (sent_id: {sent_id}):\n"{text}"')
        else:
            parts.append(f'Phân tích cảm xúc cho văn bản sau (sent_id: {sent_id}):\n"{text}"')
        return "\n".join(parts)

