"""Shared prompt resources for `src.prompt_templates`.

This module centralizes:
- System prompts per language
- Example pools (hardcoded fallback) and loading helper

It is introduced for the refactor plan while keeping legacy templates intact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS: dict[str, str] = {
    "vi": """Bạn là chuyên gia phân tích cảm xúc (Sentiment Analysis) chuyên sâu trong tiếng Việt. Nhiệm vụ của bạn là trích xuất các Opinion Tuples theo cấu trúc JSON.

1. QUY TẮC CỐT LÕI VỀ TRÍCH XUẤT (EXACT SPAN & MINIMALISM):
   - KHÔNG ĐƯỢC SUY DIỄN: Trích xuất chính xác tuyệt đối các đoạn con (substrings) từ văn bản gốc. Không sửa lỗi chính tả, không thêm từ, không bỏ bớt từ.
   - TÍNH TỐI GIẢN (MINIMALISM): Chỉ trích xuất các từ ngữ tạo thành ý nghĩa cảm xúc/đánh giá/đối tượng. 
   - ZERO EXTRACTION LÀ LỖI NGHIÊM TRỌNG: Nếu văn bản chứa từ ngữ cảm xúc, câu mỉa mai, câu hỏi tu từ, slang, teencode, hoặc emojis mang sắc thái cảm xúc, BẮT BUỘC phải trích xuất. Không được bỏ qua bất kỳ tuple nào.

2. CẤU TRÚC THÀNH PHẦN (SCHEMA):
   - SOURCE (Holder): Chủ thể phát biểu. NẾU KHÔNG CÓ ĐẠI TỪ NHÂN XƯNG TRONG VĂN BẢN, BẮT BUỘC ĐỂ TRỐNG []. Tuyệt đối không tự gán "tôi", "người viết".
   - TARGET (Đối tượng): Thực thể bị tác động hoặc được nói đến. Nếu toàn bộ mệnh đề là chủ thể của cảm xúc, hãy để Target trống [].
   - POLAR_EXPRESSION: Từ/cụm từ gốc thể hiện cảm xúc (ví dụ: "tiếc quá", "thắc mắc", "phì cười"). 
     * Lưu ý về độ dài: Đôi khi toàn bộ mệnh đề hành động chính là Polar_expression (ví dụ: "bớt chọc điên tao và bớt leo lên đầu tao ngồi"). Đừng chỉ trích xuất từ đơn lẻ nếu cụm từ mới là thực thể mang ý nghĩa cảm xúc trọn vẹn.
   - POLARITY (Positive/Negative/Neutral): Phải đánh giá dựa trên ngữ cảnh thực tế của tiếng Việt (ví dụ: "cảm ơn" đi kèm với nội dung châm biếm là Negative; các câu hỏi nghi vấn trong mỉa mai thường là Neutral hoặc Negative).

3. XỬ LÝ ĐẶC THÙ TIẾNG VIỆT:
   - PRO-DROP: Trong tiếng Việt, chủ ngữ thường bị lược bỏ. Nếu không có từ chỉ đích danh, Source luôn để trống.
   - SẮC THÁI MẠNG: 
     - Emojis (😂, :))), 🙃) là một phần của Polar_expression hoặc modifier làm thay đổi Intensity/Polarity.
     - Slang/Teencode (dm, xàm lồn, vcl...) phải được coi là Polar_expression hoặc thành phần cấu thành cảm xúc.
   - ĐA TUPLE: Một câu có thể có nhiều cặp Target-Expression độc lập. Hãy rà soát từng mệnh đề.

4. QUY TRÌNH TƯ DUY (CHAIN-OF-THOUGHT):
   Trước khi xuất JSON, thực hiện:
   - Bước 1: Chia tách câu thành các ý độc lập.
   - Bước 2: Với mỗi ý, tìm Target và Polar_expression (phải là substring từ văn bản).
   - Bước 3: Kiểm tra: "Nếu mình bỏ phần này đi, câu có mất ý nghĩa cảm xúc không?" -> Nếu không mất, hãy bỏ đi để đạt tính tối giản.
   - Bước 4: So khớp từng ký tự để đảm bảo Exact Span.

5. ĐỊNH DẠNG ĐẦU RA:
{
  "opinions": [
    {
      "Source": ["trích xuất hoặc []"],
      "Target": ["trích xuất hoặc []"],
      "Polar_expression": ["trích xuất"],
      "Polarity": "Positive/Negative/Neutral",
      "Intensity": "Strong/Standard/Weak"
    }
  ]
}

LUÔN GHI NHỚ: Độ chính xác của Span (vị trí ký tự) là quan trọng nhất. Nếu dự đoán của bạn khác với cấu trúc của ngôn ngữ tự nhiên tiếng Việt, hãy ưu tiên logic văn hóa mạng thay vì logic ngữ pháp cứng nhắc.

""",
    "en": """You are an expert in multilingual structured sentiment analysis and opinion tuple extraction. Your task is to extract Opinion Tuples from text written in any language and return them in JSON format.

1. CORE EXTRACTION RULES (EXACT SPAN & MINIMALISM):
   - NO INFERENCE ALLOWED: Extract exact substrings from the original text.
     Do not correct spelling, add words, remove words, paraphrase, normalize,
     or reinterpret the text in any way.
   - MINIMALISM: Extract only the words necessary to convey the sentiment,
     evaluation, opinion holder, or opinion target. Omit surrounding context,
     unnecessary modifiers, and filler words.
   - ZERO EXTRACTION IS A CRITICAL ERROR: If the text contains emotional
     expressions, evaluations, sarcasm, rhetorical questions, slang,
     abbreviations, internet language, or sentiment-bearing emojis, you MUST
     extract them. Do not omit any valid opinion tuple.

2. COMPONENT SCHEMA:
   - SOURCE (Holder): The speaker or opinion holder.
     * Only extract a Source when it is explicitly present as a word or phrase
       in the text. Many languages permit subject omission when the subject is
       contextually understood (pro-drop). Do not reconstruct or infer an
       implicit holder in any language.
     * If no explicit holder appears in the text, Source MUST be [].
   - TARGET: The entity, object, person, event, action, or concept being
     evaluated, discussed, or affected.
     * If no clearly identifiable target exists, Target MUST be [].
   - POLAR_EXPRESSION: The exact word, phrase, or clause expressing sentiment,
     evaluation, attitude, emotion, judgment, or opinion.
     * This field is mandatory and cannot be empty.
     * Length consideration: Sometimes an entire clause constitutes the
       Polar_expression. Do not reduce it to a single word if the full phrase
       is required to preserve the intended sentiment meaning.
   - POLARITY (Positive / Negative / Neutral):
     * Assign polarity based on the actual communicative intent of the text,
       not its surface form.
     * Account for sarcasm, irony, rhetorical questions, double negatives, and
       culturally-specific expressions. An expression that appears positive on
       the surface may carry negative sentiment in context, and vice versa.
   - INTENSITY (Strong / Standard / Weak):
     * Strong  — highly emotional, emphatic, exaggerated, or forceful.
     * Standard — ordinary, unmarked sentiment expression.
     * Weak    — mild, cautious, uncertain, or low-intensity sentiment.

3. MULTILINGUAL AND SOCIAL MEDIA CONSIDERATIONS:
   - LANGUAGE VARIATION:
     * The input text may be written in any language. All extraction rules
       apply uniformly regardless of language, script, or grammatical
       structure. Respect the pragmatic conventions of the source language
       without making inferences beyond what is explicitly stated.
   - ONLINE LANGUAGE PHENOMENA:
     * Emojis, emoticons, and graphical symbols may constitute part of a
       Polar_expression or modify Intensity and/or Polarity.
     * Slang, abbreviations, internet language, profanity, informal
       expressions, non-standard spellings, and code-switching are valid
       sentiment-bearing expressions whenever they contribute to the opinion.
   - MULTIPLE TUPLES:
     * A single sentence may contain multiple independent opinions.
       Examine every clause and proposition to identify all valid tuples.

4. ANALYSIS CHECKLIST:
   Before producing the JSON output, verify the following:
   - All opinion-bearing clauses or propositions have been identified.
   - Each opinion has a determined Source, Target, and Polar_expression.
   - Every extracted span is an exact substring of the original text.
   - No extracted span can be shortened without losing its opinion meaning.
   - Each independent opinion is represented as a separate tuple.
   - A final character-by-character check of all extracted spans has been done.

5. OUTPUT FORMAT:
{
  "opinions": [
    {
      "Source": ["extracted span or []"],
      "Target": ["extracted span or []"],
      "Polar_expression": ["extracted span"],
      "Polarity": "Positive/Negative/Neutral",
      "Intensity": "Strong/Standard/Weak"
    }
  ]
}

ALWAYS REMEMBER:
- Exact span extraction is the highest priority.
- Every extracted span must match the original text character-for-character.
- When grammatical structure conflicts with intended meaning, prioritize the
  actual meaning while preserving exact-span extraction.
- Missing a valid opinion tuple is a serious error.

""",
}


def get_system_prompt(language: str) -> str:
    """Return system prompt for a given language.

    Args:
        language: Language code ("vi", "en", ...).

    Raises:
        ValueError: If language is not supported.
    """
    if language not in SYSTEM_PROMPTS:
        raise ValueError(
            f"Unsupported language: '{language}'. Available: {sorted(SYSTEM_PROMPTS.keys())}. /
            Caution: Languages other than vi use the same system prompt as en"
        )
    else:
        if language == "vi"
          return SYSTEM_PROMPTS[language]
        else:
          print("Caution: Languages other than vi use the same system prompt as en")
          return SYSTEM_PROMPTS["en"]
          

# ---------------------------------------------------------------------------
# Hardcoded pools (fallback)
# ---------------------------------------------------------------------------
_HARDCODED_POOL: dict[str, list[dict[str, Any]]] = {
    "vi": [
        {
            "text": "hay lắm",
            "reasoning": "1. Phân tích câu: \"hay lắm\".\n2. Đối tượng (Target): Câu này không chỉ đích danh một đối tượng cụ thể nào (như \"bài hát\", \"cái này\"), do đó Target là [].\n3. Chủ thể (Source): Không có đại từ nhân xưng, Source là [].\n4. Polar_expression: \"hay lắm\" là cụm từ biểu đạt cảm xúc tích cực. \n5. Polarity: \"hay\" mang nghĩa tích cực (Positive).\n6. Intensity: \"lắm\" là từ chỉ mức độ tăng cường (Intensifier), nên Intensity là \"Strong\".",
            "output": "{\n  \"opinions\": [\n    {\n      \"Source\": [],\n      \"Target\": [],\n      \"Polar_expression\": [\"hay lắm\"],\n      \"Polarity\": \"Positive\",\n      \"Intensity\": \"Strong\"\n    }\n  ]\n}"
        },
        {
            "text": "khá tàn nhẫn",
            "reasoning": "1. Phân tích câu: \"khá tàn nhẫn\".\n2. Đối tượng (Target): Không được chỉ đích danh trong câu, nên để trống [].\n3. Chủ thể (Source): Không có đại từ nhân xưng, để trống [].\n4. Polar_expression: \"tàn nhẫn\". Từ \"khá\" đóng vai trò là trạng từ chỉ mức độ (Intensity). Do đó, toàn bộ cụm \"khá tàn nhẫn\" thể hiện thái độ đánh giá tiêu cực.\n5. Polarity: \"tàn nhẫn\" mang nghĩa tiêu cực (Negative).\n6. Intensity: \"khá\" làm giảm nhẹ cường độ của từ \"tàn nhẫn\" xuống mức Standard (thay vì Strong).",
            "output": "{\n  \"opinions\": [\n    {\n      \"Source\": [],\n      \"Target\": [],\n      \"Polar_expression\": [\"khá tàn nhẫn\"],\n      \"Polarity\": \"Negative\",\n      \"Intensity\": \"Standard\"\n    }\n  ]\n}"
        },
        {
            "text": "đéo hiểu",
            "reasoning": "1. Câu \"đéo hiểu\" là một câu cảm thán mang tính phủ định mạnh mẽ về mặt sắc thái, thể hiện sự bực bội hoặc không đồng tình với một sự việc/đối tượng nào đó.\n2. Source: Không có chủ thể cụ thể, để trống [].\n3. Target: Đối tượng bị nói đến không xuất hiện cụ thể trong câu, để trống [].\n4. Polar_expression: Cụm từ \"đéo hiểu\" mang sắc thái tiêu cực mạnh.\n5. Polarity: Negative.\n6. Intensity: Strong (do sử dụng từ chửi thề \"đéo\").",
            "output": "{\n  \"opinions\": [\n    {\n      \"Source\": [],\n      \"Target\": [],\n      \"Polar_expression\": [\"đéo hiểu\"],\n      \"Polarity\": \"Negative\",\n      \"Intensity\": \"Strong\"\n    }\n  ]\n}"
        },
        {
            "text": "per thanh niên suốt ngày hát karaoke max volum",
            "reasoning": "1. Câu văn \"per thanh niên suốt ngày hát karaoke max volum\" là một lời phàn nàn của người viết về một đối tượng cụ thể.\n2. Target: \"thanh niên\" (người bị chỉ trích).\n3. Polar_expression: \"suốt ngày hát karaoke max volum\". Đây là cụm từ diễn tả hành vi gây khó chịu, mang tính chất tiêu cực.\n4. Source: Không có đại từ nhân xưng chỉ người nói, để trống [].\n5. Polarity: Negative (vì thể hiện sự bức xúc trước hành vi làm phiền).\n6. Intensity: Standard (lời than phiền về hành động gây ồn ào).",
            "output": "{\n  \"opinions\": [\n    {\n      \"Source\": [],\n      \"Target\": [\"thanh niên\"],\n      \"Polar_expression\": [\"suốt ngày hát karaoke max volum\"],\n      \"Polarity\": \"Negative\",\n      \"Intensity\": \"Standard\"\n    }\n  ]\n}"
        },
        {
            "text": "sợ nhất con quỷ này trên đời",
            "reasoning": "1. Phân tích câu: \"sợ nhất con quỷ này trên đời\".\n2. Xác định chủ thể cảm xúc: Không có đại từ nhân xưng, Source để trống [].\n3. Xác định đối tượng (Target): \"con quỷ này\" là đối tượng bị đánh giá.\n4. Xác định cụm biểu đạt cảm xúc (Polar_expression): \"sợ nhất\" thể hiện mức độ cảm xúc mạnh mẽ đối với đối tượng.\n5. Đánh giá cực tính (Polarity): \"sợ\" mang sắc thái tiêu cực (Negative).\n6. Đánh giá cường độ (Intensity): \"nhất\" thể hiện mức độ cao nhất (Strong).\n7. Kiểm tra tính tối giản: \"sợ nhất\" và \"con quỷ này\" là các thành phần cốt lõi tạo nên ý nghĩa cảm xúc.",
            "output": "{\n  \"opinions\": [\n    {\n      \"Source\": [],\n      \"Target\": [\"con quỷ này\"],\n      \"Polar_expression\": [\"sợ nhất\"],\n      \"Polarity\": \"Negative\",\n      \"Intensity\": \"Strong\"\n    }\n  ]\n}"
        }
    ],  
    "en": [],
    "es": [],
    "nor": [],
    "ca": [],
    "eu": [],  
}


def load_examples_pool(path: Optional[str], language: str) -> list[dict[str, Any]]:
    """Load examples pool from JSON path or fall back to hardcoded pool.

    Args:
        path: JSON file path (list of dict examples). If None, use hardcoded pool.
        language: Language code ("vi", "en", ...).
    """
    if path:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Examples pool JSON must be a list, got: {type(data)}")
        return data
    return list(_HARDCODED_POOL.get(language, []))

