"""Pure prompt-building blocks for `src.prompt_templates`.

These functions are intentionally stateless and deterministic: they accept
data and return strings. They should not read files or maintain global state.

Post-reorg note: this module used to back a generic `PromptCreator` /
`ContentProvider` pipeline (`contents.py`, `creator.py`, `factory.py`) that
built entire multi-example prompt blocks in one call (e.g. `few_shot_block`
rendering N examples at once). That pipeline has been removed — every
technique (`few_shot`, `re_reading`, `few_shot_cot`, `plan_and_solve`) now
goes through lm-eval-harness's native per-technique yaml + `fewshot_config`,
which renders ONE example at a time via `doc_to_text`/`doc_to_target`
callables in `../tasks/vietnamese_ssa/utils.py`. Only the functions still
called from there remain here:
    - pas_instruction_block   -> plan_and_solve
    - base_question_block     -> few_shot (eval question) / cot (base, via cot_doc_to_text)
    - rereading_block         -> re_reading
    - simplify_opinions       -> few_shot / re_reading fewshot target formatting
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Plan-and-Solve instruction strings (copied from legacy PlanAndSolvePrompt)
# ---------------------------------------------------------------------------
PAS_INSTRUCTION_VI = """
Đầu tiên hãy hiểu vấn đề và vạch ra kế hoạch để giải quyết. 
Sau đó, hãy thực hiện kế hoạch, phân tích từng bước và đưa ra câu trả lời cuối cùng. 
Vui lòng sinh ra quy trình cụ thể theo các bước: 
[Hiểu vấn đề], [Lập kế hoạch], [Phân tích chi tiết], [Câu trả lời].
""".strip()

PAS_PLUS_INSTRUCTION_VI = """
Đầu tiên hãy hiểu vấn đề, trích xuất các thành phần chủ thể, đối tượng, biểu thức cảm xúc của từng ý kiến (opinion), và vạch ra kế hoạch để giải quyết.
Sau đó, hãy thực hiện kế hoạch, phân tích từng bước, thiết lập các mối liên kết giữa các thành phần, phân định nhãn cực tính 
(lưu ý kỹ đến tính chính xác trong xác định liên kết giữa các thành phần và các yếu tố ngữ cảnh mỉa mai) và đưa ra câu trả lời cuối cùng.
Vui lòng sinh ra quy trình cụ thể theo các bước: 
[Hiểu vấn đề], [Lập kế hoạch], [Phân tích chi tiết], [Câu trả lời].
""".strip()

PAS_INSTRUCTION_EN = """
Let's first understand the problem and devise a plan to solve the problem. 
Then, let's carry out the plan, solve the problem step by step and give the ultimate answer.
Please explicitly generate the mentioned process: 
[Problem Understanding], [Plan], [Detailed Analysis], [Answer].
""".strip()

PAS_PLUS_INSTRUCTION_EN = """Perform a detailed analysis following these steps:
1. **Preliminary Analysis & Signal Extraction:** Read the text carefully. Extract all keywords, personal pronouns, slang (teencode), and linguistic signals that might relate to sentiment.
2. **Analysis Planning:** Based on the extracted signals, outline a plan to sequentially analyze each Opinion.
3. **Plan Execution:**
    - For each Opinion, determine the value for each component (Source, Target, Polar_expression, Polarity).
    - Briefly explain your reasoning, paying special attention to context, implicit meaning, and the defined rules.
4. **Result Synthesis:** Construct the final JSON block based on the entire analysis above.
""".strip()


def pas_instruction_block(*, plus: bool = False, language: str = "vi") -> str:
    """Return Plan-and-Solve instruction block."""
    if language == "en":
        return PAS_PLUS_INSTRUCTION_EN if plus else PAS_INSTRUCTION_EN
    return PAS_PLUS_INSTRUCTION_VI if plus else PAS_INSTRUCTION_VI


# ---------------------------------------------------------------------------
# Shared opinion-JSON simplification (used by fewshot target rendering)
# ---------------------------------------------------------------------------
def simplify_opinions(opinions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    simplified: list[dict[str, Any]] = []
    for op in opinions:
        def extract_text(field_name: str) -> list[str]:
            val = op.get(field_name)
            if not val:  # Xử lý các giá trị None hoặc [] rỗng
                return []
            if isinstance(val, list):
                if len(val) > 0:
                    first_el = val[0]
                    # Nếu thuộc định dạng SemEval: [["văn bản"], [vị trí]]
                    if isinstance(first_el, list):
                        return first_el
                    # Nếu đã ở định dạng danh sách phẳng: ["văn bản"]
                    return val
            return []

        simplified.append(
            {
                "Source": extract_text("Source"),
                "Target": extract_text("Target"),
                "Polar_expression": extract_text("Polar_expression"),
                "Polarity": op.get("Polarity", ""),
            }
        )
    return simplified


# ---------------------------------------------------------------------------
# Base question block (copied from legacy user prompts)
# ---------------------------------------------------------------------------
def base_question_block(text: str, sent_id: str, language: str = "vi") -> str:
    """Base question block used in most single-stage techniques."""
    if language == "en":
        return (
            f"""Analyze the sentiment for the following text:
"{text}"
"""
        )
    return (
        f"""Phân tích cảm xúc cho văn bản sau:
"{text}"
"""
    )


# ---------------------------------------------------------------------------
# Re-reading block (copied from legacy ReReadingPrompt core formatting)
# ---------------------------------------------------------------------------
def rereading_block(text: str, sent_id: str, language: str = "vi") -> str:
    """Wrap base question with Re-reading instruction."""
    if language == "en":
        base_q = f"""Analyze the sentiment for the following text:
"{text}"
"""
        reread = (
            f"""Read the question again: {base_q}"""
        )
        return f"{base_q}\n\n{reread}"

    base_q = f"""Phân tích cảm xúc cho văn bản sau:
"{text}"
"""
    reread = (
        f"""Đọc lại câu hỏi: {base_q}"""
    )
    return f"{base_q}\n\n{reread}"