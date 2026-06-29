"""Pure prompt-building blocks for `src.prompt_templates`.

These functions are intentionally stateless and deterministic: they accept
data and return strings. They should not read files or maintain global state.

The block formats are copied from the legacy templates to ensure the new
refactor path remains compatible with lm-eval runs.
"""

from __future__ import annotations

import json
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
# Few-shot formatting (copied from legacy FewShotPrompt)
# ---------------------------------------------------------------------------
def _simplify_opinions(opinions: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def few_shot_block(examples: list[dict[str, Any]], language: str = "vi") -> str:
    """Build few-shot examples block (no reasoning)."""
    if language == "en":
        section = "Here are some examples:\n\n"
        label = "Example"
    else:
        section = "Dưới đây là một số ví dụ:\n\n"
        label = "Ví dụ"

    for i, ex in enumerate(examples, 1):
        formatted_output = {
            "opinions": _simplify_opinions(ex.get("opinions", []))
        }
        section += f"{label} {i}:\n"
        section += f'Input: "{ex.get("text", "")}"\n'
        section += f"Output: {json.dumps(formatted_output, ensure_ascii=False, indent=2)}"
        section += "\n\n"

    return section.strip()


# ---------------------------------------------------------------------------
# CoT demonstration formatting (copied from legacy FewShotCoTPrompt)
# ---------------------------------------------------------------------------
def cot_demo_block(examples: list[dict[str, Any]], language: str = "vi") -> str:
    """Build demonstration block with reasoning + output JSON."""
    if language == "en":
        section = "HERE ARE SOME DEMONSTRATION EXAMPLES:\n\n"
        ex_label = "EXAMPLE"
    else:
        section = "DƯỚI ĐÂY LÀ MỘT SỐ VÍ DỤ MINH HỌA:\n\n"
        ex_label = "VÍ DỤ"

    for i, ex in enumerate(examples, 1):
        section += f"=== {ex_label} {i} ===\n"
        section += f'Input: "{ex.get("text", "")}"\n\n'
        section += f"Reasoning:\n{ex.get('reasoning', '')}\n\n"
        section += "Output:\n"
        
        output_data = ex.get("output", {})
        if isinstance(output_data, str):
            section += output_data.strip()
        else:
            section += json.dumps(output_data, ensure_ascii=False, indent=2)
        section += "\n\n"

    return section.strip()


# ---------------------------------------------------------------------------
# Base question blocks (copied from legacy user prompts)
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
        f"""Phân tích cảm xúc cho văn bản sau):
"{text}"
"""
    )


def few_shot_question_block(text: str, sent_id: str, language: str = "vi") -> str:
    """Question block used by legacy FewShotPrompt when examples exist."""
    if language == "en":
        return (
            f"""Now, analyze the sentiment for the following text):
Input: "{text}"
Output:
"""
        )
    return (
        f"""Bây giờ, phân tích cảm xúc cho văn bản sau:
Input: "{text}"
Output:
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


# ---------------------------------------------------------------------------
# Re-reading example formatting (copied from legacy ReReadingPrompt)
# ---------------------------------------------------------------------------
def re2_examples_block(
    examples: list[dict[str, Any]],
    *,
    include_reasoning: bool = False,
    language: str = "vi",
) -> str:
    """Build RE2-style examples section where each example includes re-reading."""
    if language == "en":
        section = "HERE ARE SOME DEMONSTRATION EXAMPLES:\n\n"
        header = "=== EXAMPLE"
        for i, ex in enumerate(examples, 1):
            text = ex.get("text", "")
            q1 = f'Analyze the sentiment for the following text: "{text}"'
            q2 = f"Read the question again: {q1}"
            section += f"{header} {i} ===\n"
            section += f"{q1}\n\n{q2}\n\n"
            if include_reasoning and "reasoning" in ex:
                section += f"Reasoning:\n{ex.get('reasoning','')}\n\n"
            if include_reasoning:
                output_data = ex.get("output", {})
            else:
                output_data = {"text": text, "opinions": _simplify_opinions(ex.get("opinions", []))}
            section += f"Output:\n{json.dumps(output_data, ensure_ascii=False, indent=2)}\n\n"
        return section.strip()

    section = "DƯỚI ĐÂY LÀ MỘT SỐ VÍ DỤ MINH HỌA:\n\n"
    header = "=== VÍ DỤ"
    for i, ex in enumerate(examples, 1):
        text = ex.get("text", "")
        q1 = f'Phân tích cảm xúc cho văn bản sau: "{text}"'
        q2 = f"Đọc lại câu hỏi: {q1}"
        section += f"{header} {i} ===\n"
        section += f"{q1}\n\n{q2}\n\n"
        if include_reasoning and "reasoning" in ex:
            section += f"Reasoning:\n{ex.get('reasoning','')}\n\n"
        if include_reasoning:
            output_data = ex.get("output", {})
        else:
            output_data = {"text": text, "opinions": _simplify_opinions(ex.get("opinions", []))}
        section += f"Output:\n{json.dumps(output_data, ensure_ascii=False, indent=2)}\n\n"
    return section.strip()

