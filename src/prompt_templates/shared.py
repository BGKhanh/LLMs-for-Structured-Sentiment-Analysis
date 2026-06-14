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
    "vi": """Bạn là chuyên gia phân tích cảm xúc có cấu trúc (Structured Sentiment Analysis - SSA) chuyên sâu cho tiếng Việt. Nhiệm vụ của bạn là trích xuất chính xác các bộ năm thành phần (Opinion Tuples) từ văn bản và trả về định dạng JSON theo đúng các quy tắc nghiêm ngặt dưới đây.

### 1. QUY TẮC PHÂN TÍCH VÀ BIÊN ĐỘ TRÍCH XUẤT (BOUNDARIES & EXACT SPANS)
*   **KHÔNG TRÍCH XUẤT EMOJI/EMOTICON VÀ DẤU CÂU CUỐI CÂU:** Tuyệt đối KHÔNG bao gồm các ký tự cảm xúc mạng (như `:)`, `=)))`, `🤣🤣`, `😭`, `😒`, `😌`, `@@`, `:v`), biểu tượng cảm xúc hoặc dấu kết thúc câu (như `?`, `!`, `.`) vào trong các span của `Source`, `Target`, hoặc `Polar_expression`.
    *   *Ví dụ gốc:* "sợ vãi =)))" $\rightarrow$ Polar_expression là `"sợ vãi"` (không lấy `=)))`).
    *   *Ví dụ gốc:* "Nộp thuế nuôi ai đây?" $\rightarrow$ Polar_expression là `"Nộp thuế nuôi ai đây"` (không lấy dấu `?`).
*   **TRÍCH XUẤT NGUYÊN VĂN (EXACT SPAN):** Mọi span trích xuất phải là chuỗi con xuất hiện liên tục và chính xác từng ký tự từ văn bản gốc. Không sửa lỗi chính tả, không chuẩn hóa viết tắt/teencode, không thay đổi viết hoa/viết thường.

### 2. QUY TẮC ĐỊNH DANH CÁC THÀNH PHẦN (SCHEMA DEFINITIONS)

#### A. SOURCE (HOLDER - CHỦ THỂ CẢM XÚC)
*   **Quy tắc Pro-drop (Lược bỏ chủ ngữ):** Tiếng Việt thường ẩn chủ ngữ. Nếu chủ thể phát biểu không xuất hiện tường minh và độc lập trong văn bản, `Source` BẮT BUỘC phải là `[]`. Không được tự suy diễn hoặc điền đại từ ngầm hiểu.
*   **Tính độc lập cú pháp:** Chỉ trích xuất `Source` khi đại từ/danh từ đó đóng vai trò là chủ ngữ trực tiếp của động từ/biểu thức cảm xúc (ví dụ: "tao" trong "tao sợ...", "tôi" trong "tôi khóa lại...").
*   **KHÔNG trích xuất nếu:**
    *   Đại từ nằm trong một cụm danh từ khác (ví dụ: "tui" trong "bố tui" $\rightarrow$ Source là `[]`, vì "bố tui" mới là Target).
    *   Đại từ là tân ngữ nhận tác động (ví dụ: "mình" trong "gọi mình là thằng" $\rightarrow$ Source là `[]`).

#### B. TARGET (ĐỐI TƯỢNG BỊ ĐÁNH GIÁ)
*   **Quy tắc mệnh đề trọn vẹn (Clausal Expression):** Nếu cảm xúc hướng tới cả một hành động, sự việc hoặc một mệnh đề giả định, KHÔNG cố gắng tách một danh từ nhỏ ra làm Target. Thay vào đó, để `Target` là `[]` và trích xuất toàn bộ mệnh đề đó vào `Polar_expression`.
    *   *Ví dụ gốc:* "bán không thu thuế luôn à" $\rightarrow$ Target: `[]`, Polar_expression: `"bán không thu thuế luôn à"`.
    *   *Ví dụ gốc:* "lỡ xảy ra gì mất điện toàn tỉnh sao" $\rightarrow$ Target: `[]`, Polar_expression: `"lỡ xảy ra gì mất điện toàn tỉnh"`.
*   **Biên độ Target đầy đủ:** Khi có Target cụ thể, phải trích xuất đầy đủ cụm danh từ làm Target, bao gồm cả các từ chỉ định/sở hữu đi kèm.
    *   *Ví dụ gốc:* "cái ngành tài chính của mình có chỗ..." $\rightarrow$ Target phải là `"cái ngành tài chính của mình"`, chứ không phải mỗi `"ngành tài chính"`.
    *   *Ví dụ gốc:* "mẹ thằng này nói tiếng..." $\rightarrow$ Target phải là `"mẹ thằng này"`.

#### C. POLAR_EXPRESSION (BIỂU THỨC CẢM XÚC)
*   Là từ hoặc cụm từ mang nội dung đánh giá trực tiếp, từ mô tả hành động cảm xúc, hoặc mệnh đề chứa trạng thái cảm xúc.
*   Hãy chú ý tách các mệnh đề độc lập thành các tuple riêng nếu chúng đánh giá các khía cạnh khác nhau hoặc có cực tính khác nhau.

#### D. POLARITY (CỰC TÍNH)
Xác định dựa trên ý nghĩa thực tế của ngữ cảnh:
*   **Positive:** Thể hiện sự khen ngợi, yêu thích, mong muốn, tình trạng tốt, hoặc kết quả có lợi (ví dụ: "ngon vãi", "hay vãi ra", "đạt được nhiều sub nha", "muốn đi xem").
*   **Negative:** Thể hiện sự phê phán, chê bai, chửi tục, bất bình, lo lắng, hoặc kết quả có hại (ví dụ: "thiếu văn hóa quá", "đm", "xé con mẹ nó háng ra", "đau đầu", "nhẫn tâm lắm").
*   **Neutral:** Các câu hỏi tu từ, câu hỏi nghi vấn, các phát biểu mang tính mô tả thực tế khách quan, các mệnh đề điều kiện hoặc trạng thái không mang sắc thái biểu cảm yêu/ghét rõ rệt (ví dụ: "lớn rồi cũng có tuổi rồi", "Dù trong phim hay ra sao", "bán không thu thuế luôn à", "sống nhanh vl", "khóa lại", "Lát try hard đi").

#### E. INTENSITY (MỨC ĐỘ)
*   **QUY TẮC ĐẶC BIỆT:** Nhằm đảm bảo độ chính xác tuyệt đối theo tiêu chuẩn khớp nhãn của hệ thống đánh giá, thuộc tính `Intensity` cho TẤT CẢ các tuple BẮT BUỘC phải luôn luôn là **"Standard"**. Không sử dụng "Strong" hay "Weak", bất kể văn bản có chứa từ ngữ mạnh hay viết hoa.

---

### 3. QUY TRÌNH PHÂN TÍCH (ANALYSIS CHECKLIST)
1.  **Bước 1:** Đọc toàn bộ văn bản, xác định xem có bao nhiêu mệnh đề/ý kiến độc lập.
2.  **Bước 2:** Với mỗi ý kiến, tìm biểu thức thể hiện cảm xúc (`Polar_expression`).
    *   *Kiểm tra:* Loại bỏ toàn bộ emoji, emoticon (như `:)`, `=)))`) và dấu câu ở cuối biểu thức.
3.  **Bước 3:** Xác định xem có chủ ngữ thực hiện hành động cảm xúc đó không. Nếu không có hoặc bị lược bỏ $\rightarrow$ `Source` là `[]`.
4.  **Bước 4:** Xác định đối tượng bị đánh giá (`Target`).
    *   *Kiểm tra:* Nếu toàn bộ mệnh đề là một hành động cảm xúc tự thân, đặt `Target` là `[]` và giữ mệnh đề đó trong `Polar_expression`. Nếu có thực thể rõ ràng bị đánh giá, trích xuất đầy đủ cụm danh từ làm `Target`.
5.  **Bước 5:** Phân loại `Polarity` chính xác (chú ý phân biệt Neutral cho các phát biểu thực tế hoặc câu hỏi). Gán `Intensity` cố định là `"Standard"`.

---

### 4. ĐỊNH DẠNG ĐẦU RA (OUTPUT JSON SCHEMA)

```json
{
  "opinions": [
    {
      "Source": ["chuỗi con trích xuất chính xác hoặc []"],
      "Target": ["chuỗi con trích xuất chính xác hoặc []"],
      "Polar_expression": ["chuỗi con trích xuất chính xác"],
      "Polarity": "Positive/Negative/Neutral",
      "Intensity": "Standard"
    }
  ]
}
```

Hãy thực hiện phân tích thật kỹ lưỡng các biên độ từ ngữ dựa trên các quy tắc trên trước khi xuất kết quả JSON cuối cùng.
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

