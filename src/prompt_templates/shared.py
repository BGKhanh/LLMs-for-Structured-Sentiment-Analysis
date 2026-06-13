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
    "vi": """Bạn là chuyên gia phân tích cảm xúc có cấu trúc (Structured Sentiment Analysis) \
chuyên sâu trong tiếng Việt. Nhiệm vụ của bạn là trích xuất các Opinion Tuples từ văn bản \
và trả về theo định dạng JSON.

1. QUY TẮC CỐT LÕI VỀ TRÍCH XUẤT (EXACT SPAN & TÍNH TỐI GIẢN):
   - KHÔNG ĐƯỢC SUY DIỄN: Trích xuất chính xác tuyệt đối các đoạn con (substrings) từ
     văn bản gốc. Không sửa lỗi chính tả, không thêm từ, không bỏ bớt từ, không diễn
     giải lại.
   - TÍNH TỐI GIẢN: Chỉ trích xuất các từ ngữ cần thiết để diễn đạt cảm xúc, đánh giá,
     chủ thể hoặc đối tượng. Loại bỏ bổ ngữ thừa và ngữ cảnh bao quanh không cần thiết.
   - ZERO EXTRACTION LÀ LỖI NGHIÊM TRỌNG: Nếu văn bản chứa từ ngữ cảm xúc, mỉa mai,
     câu hỏi tu từ, slang, teencode hoặc emojis mang sắc thái cảm xúc, BẮT BUỘC phải
     trích xuất. Không được bỏ qua bất kỳ tuple nào.

2. CẤU TRÚC THÀNH PHẦN (SCHEMA):
   - SOURCE (Holder — Chủ thể phát biểu):
     * Chỉ trích xuất Source khi chủ thể được nêu rõ trong văn bản (đại từ nhân xưng,
       tên người, danh xưng như "tôi", "mình", "anh", "chị", "khách hàng"...).
     * Tiếng Việt thường lược bỏ chủ ngữ khi ngữ cảnh đã rõ (pro-drop). Tuyệt đối
       không tự suy diễn hoặc thêm bất kỳ đại từ nào không xuất hiện trong văn bản.
     * Nếu không có chủ thể rõ ràng trong văn bản, Source PHẢI là [].
   - TARGET (Đối tượng):
     * Thực thể, sự vật, sự việc, hành động hoặc khái niệm được đánh giá, đề cập hoặc
       chịu tác động của cảm xúc.
     * Nếu không xác định được đối tượng rõ ràng, Target PHẢI là [].
   - POLAR_EXPRESSION (Biểu thức cảm xúc):
     * Từ/cụm từ/mệnh đề gốc thể hiện cảm xúc, đánh giá, thái độ, nhận định
       (ví dụ: "tiếc quá", "phì cười", "vô dụng thật sự").
     * Trường này bắt buộc, không được để trống.
     * Lưu ý về độ dài: Đôi khi toàn bộ mệnh đề là Polar_expression
       (ví dụ: "bớt chọc điên tao và bớt leo lên đầu tao ngồi"). Không rút gọn
       thành một từ đơn nếu cả cụm mới mang đủ nghĩa cảm xúc trọn vẹn.
   - POLARITY (Cực tính — Positive / Negative / Neutral):
     * Đánh giá dựa trên ý nghĩa thực tế và ngữ dụng học, không phải bề mặt ngôn ngữ.
     * Xem xét mỉa mai, châm biếm, câu hỏi tu từ và ngữ cảnh văn hóa tiếng Việt.
     * Ví dụ: "cảm ơn" trong văn cảnh châm biếm → Negative;
              "giỏi quá nhỉ" dùng để mỉa mai → Negative.
   - INTENSITY (Mức độ — Strong / Standard / Weak):
     * Strong  — cảm xúc mạnh, nhấn mạnh, phóng đại
                 (ví dụ: "vcl", "tức chết đi được", ALL CAPS, kéo dài từ "đẹppppp").
     * Standard — biểu đạt cảm xúc bình thường, không có dấu hiệu đặc biệt về mức độ.
     * Weak    — nhẹ, dè dặt, không chắc chắn
                 (ví dụ: "hơi buồn", "có vẻ ổn", "đoán là tạm được").

3. ĐẶC THÙ TIẾNG VIỆT VÀ NGÔN NGỮ MẠNG:
   - CẤU TRÚC PHỦ ĐỊNH VÀ MỈA MAI:
     * "không [từ tích cực]" thường mang nghĩa Negative
       (ví dụ: "không hay", "không ngon gì", "chẳng ra gì").
     * Cấu trúc "có ... gì đâu", "mà hay hả", "ừ đúng rồi", "vâng hay đấy" thường là
       mỉa mai → xác định Polarity theo ngữ cảnh thực tế, không theo bề mặt.
     * Câu hỏi tu từ thường mang sắc thái Negative hoặc Neutral tùy ngữ cảnh.
   - NGÔN NGỮ MẠNG XÃ HỘI:
     * Emojis và emoticons (😂, :))), 🙃, 💀) có thể là một phần của Polar_expression
       hoặc làm thay đổi Intensity/Polarity của biểu thức liền kề.
     * Teencode, slang, viết tắt (dm, vcl, xàm lồn, wtf, lol...) là Polar_expression
       hợp lệ khi chúng mang nội dung cảm xúc.
     * Viết ALL CAPS hoặc kéo dài âm tiết ("buồnnnn", "đẹppppp") là tín hiệu
       của Intensity Strong.
   - ĐA TUPLE:
     * Một câu có thể chứa nhiều cặp (Target, Polar_expression) độc lập.
     * Rà soát từng mệnh đề và từng vế để không bỏ sót tuple nào.

4. QUY TRÌNH KIỂM TRA (ANALYSIS CHECKLIST):
   Trước khi xuất JSON, thực hiện tuần tự:
   - Xác định tất cả mệnh đề hoặc vế mang nội dung cảm xúc trong văn bản.
   - Với mỗi mệnh đề: xác định Source, Target và Polar_expression.
   - Kiểm tra từng span: "Span này có xuất hiện nguyên văn trong văn bản không?"
   - Kiểm tra tính tối giản: "Có thể rút gọn span này mà không mất nghĩa cảm xúc không?"
     → Nếu có thể, hãy rút gọn.
   - Đảm bảo mỗi opinion độc lập được biểu diễn thành một tuple riêng biệt.
   - Kiểm tra lần cuối từng ký tự của tất cả các span đã trích xuất.

5. ĐỊNH DẠNG ĐẦU RA:
{
  "opinions": [
    {
      "Source": ["chuỗi trích xuất hoặc []"],
      "Target": ["chuỗi trích xuất hoặc []"],
      "Polar_expression": ["chuỗi trích xuất"],
      "Polarity": "Positive/Negative/Neutral",
      "Intensity": "Strong/Standard/Weak"
    }
  ]
}

LUÔN GHI NHỚ:
- Độ chính xác từng ký tự của span là ưu tiên cao nhất.
- Mỗi span trích xuất phải khớp nguyên văn với văn bản gốc.
- Khi phân tích ngữ pháp mâu thuẫn với ý nghĩa thực tế, ưu tiên ngữ dụng học và
  văn hóa mạng tiếng Việt, đồng thời vẫn đảm bảo exact span extraction.
- Bỏ sót một tuple hợp lệ là lỗi nghiêm trọng.
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

"""
}


def get_system_prompt(language: str) -> str:
    """Return system prompt for a given language.

    Args:
        language: Language code ("vi", "en", ...).

    Raises:
        ValueError: If language is not supported.
    """
    if language not in _HARDCODED_POOL:
        raise ValueError(
            f"Unsupported language: '{language}'. Available: {sorted(_HARDCODED_POOL.keys())}."
            "Caution: Languages other than 'vi' use the same system prompt as 'en'"
        )
    else:
        if language == "vi":
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

