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

---

### 3. QUY TRÌNH PHÂN TÍCH (ANALYSIS CHECKLIST)
1.  **Bước 1:** Đọc toàn bộ văn bản, xác định xem có bao nhiêu mệnh đề/ý kiến độc lập.
2.  **Bước 2:** Với mỗi ý kiến, tìm biểu thức thể hiện cảm xúc (`Polar_expression`).
    *   *Kiểm tra:* Loại bỏ toàn bộ emoji, emoticon (như `:)`, `=)))`) và dấu câu ở cuối biểu thức.
3.  **Bước 3:** Xác định xem có chủ ngữ thực hiện hành động cảm xúc đó không. Nếu không có hoặc bị lược bỏ $\rightarrow$ `Source` là `[]`.
4.  **Bước 4:** Xác định đối tượng bị đánh giá (`Target`).
    *   *Kiểm tra:* Nếu toàn bộ mệnh đề là một hành động cảm xúc tự thân, đặt `Target` là `[]` và giữ mệnh đề đó trong `Polar_expression`. Nếu có thực thể rõ ràng bị đánh giá, trích xuất đầy đủ cụm danh từ làm `Target`.
5.  **Bước 5:** Phân loại `Polarity` chính xác (chú ý phân biệt Neutral cho các phát biểu thực tế hoặc câu hỏi).
---

### 4. ĐỊNH DẠNG ĐẦU RA (OUTPUT JSON SCHEMA)

```json
{
  "opinions": [
    {
      "Source": ["chuỗi con trích xuất chính xác hoặc []"],
      "Target": ["chuỗi con trích xuất chính xác hoặc []"],
      "Polar_expression": ["chuỗi con trích xuất chính xác"],
      "Polarity": "Positive/Negative/Neutral"
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

3. MULTILINGUAL AND SOCIAL MEDIA CONSIDERATIONS:
   - LANGUAGE VARIATION:
     * The input text may be written in any language. All extraction rules
       apply uniformly regardless of language, script, or grammatical
       structure. Respect the pragmatic conventions of the source language
       without making inferences beyond what is explicitly stated.
   - ONLINE LANGUAGE PHENOMENA:
     * Emojis, emoticons, and graphical symbols may constitute part of a
       Polar_expression and/or modify Polarity.
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
      "Polarity": "Positive/Negative/Neutral"
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
    if language not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: '{language}'. Available: {sorted(_SUPPORTED_LANGUAGES)}. "
            "Caution: Languages other than 'vi' use the same system prompt as 'en'"
        )
    if language == "vi":
        return SYSTEM_PROMPTS["vi"]
    print("Caution: Languages other than vi use the same system prompt as en")
    return SYSTEM_PROMPTS["en"]
    
          
DATASET_LANGUAGES: dict[str, str] = {
    "vitoed": "vi",
    "opener_en": "en",
    "mpqa": "en",
    "darmstadt_unis": "en",
    "opener_es": "es",
    "norec": "nor",
    "multibooked_eu": "eu",
    "multibooked_ca": "ca",
}


_SUPPORTED_LANGUAGES = set(DATASET_LANGUAGES.values())


def get_language(dataset: str) -> str:
    """Resolve the language code for a given dataset name.

    Raises:
        ValueError: If the dataset isn't registered in `DATASET_LANGUAGES`.
    """
    try:
        return DATASET_LANGUAGES[dataset]
    except KeyError:
        raise ValueError(
            f"Unsupported dataset: '{dataset}'. Available: {sorted(DATASET_LANGUAGES.keys())}"
        ) from None
        
# ---------------------------------------------------------------------------
# Hardcoded pools (fallback)
# ---------------------------------------------------------------------------
_HARDCODED_POOL: dict[str, list[dict[str, Any]]] = {
    "vitoed": [
        {
            "text": "bởi nói để trung nguyên cho bà thảo thì chắc chắn 100 % với cái bộ óc sáng tạo cùng với sự lãnh đạo tài giỏi và lòng tham hơn người và trí tuệ ngắn hạn sẽ dẫn dắt trung nguyên xuống con mẹ nó giếng luôn .",
            "reasoning": """
Bước 1: Xác định số mệnh đề/ý kiến độc lập
Câu này là một câu phức, gồm nhiều cụm đánh giá nối với nhau bằng "cùng với", "và":

Cụm 1: "bộ óc sáng tạo cùng với sự lãnh đạo tài giỏi" → mang sắc thái khen (năng lực tốt)
Cụm 2: "lòng tham hơn người và trí tuệ ngắn hạn" → mang sắc thái chê (phẩm chất xấu)
Cụm 3: "sẽ dẫn dắt trung nguyên xuống con mẹ nó giếng luôn" → mệnh đề kết quả/hệ quả tiêu cực

→ Vì các cụm này có cực tính (polarity) khác nhau và đánh giá các khía cạnh khác nhau (năng lực tốt vs. phẩm chất xấu vs. hậu quả), theo quy tắc Mục C ("tách các mệnh đề độc lập thành các tuple riêng nếu chúng đánh giá các khía cạnh khác nhau hoặc có cực tính khác nhau"), ta tách thành 3 tuple riêng biệt.

Bước 2: Xác định Polar_expression cho từng ý kiến (loại bỏ dấu câu cuối)
Tuple A: "bộ óc sáng tạo cùng với sự lãnh đạo tài giỏi" — trích nguyên văn cụm danh từ ghép thể hiện năng lực.
Tuple B: "lòng tham hơn người và trí tuệ ngắn hạn" — trích nguyên văn cụm danh từ ghép thể hiện phẩm chất tiêu cực.
Tuple C: "sẽ dẫn dắt trung nguyên xuống con mẹ nó giếng luôn" — đây là mệnh đề kết quả mang tính hành động ẩn dụ ("xuống giếng" = thành ngữ chỉ sự sụp đổ/diệt vong), cần giữ nguyên cả mệnh đề chứ không tách nhỏ. Bỏ dấu chấm . ở cuối câu gốc.

Bước 3: Xác định Source cho từng tuple
Toàn câu không có đại từ nhân xưng hay danh từ nào đóng vai trò chủ ngữ phát ngôn độc lập (không có "tao nghĩ", "tôi thấy"...). Cấu trúc "bởi nói để... thì chắc chắn 100%..." là lối nói phiếm chỉ, không có chủ thể tường minh.
→ Theo quy tắc Pro-drop (Mục A): Source = [] cho cả 3 tuple.

Bước 4: Xác định Target cho từng tuple
Xét cụm "với cái bộ óc sáng tạo... và trí tuệ ngắn hạn sẽ dẫn dắt trung nguyên...": về mặt cú pháp, các phẩm chất (bộ óc, lòng tham, trí tuệ...) là phẩm chất thuộc về "bà thảo" — người được giao Trung Nguyên ("để trung nguyên cho bà thảo"). Đây không phải trường hợp "mệnh đề trọn vẹn không có target cụ thể" vì có một thực thể rõ ràng đang được bàn tới: bà thảo.
Theo quy tắc Biên độ Target đầy đủ (Mục B): Target phải trích đủ cụm danh từ chỉ người — ở đây là "bà thảo".
Áp dụng cho cả 3 tuple, vì cả 3 phẩm chất/hành động (bộ óc sáng tạo, lòng tham, hành động dẫn dắt Trung Nguyên xuống giếng) đều quy về cùng một chủ thể bị đánh giá là bà thảo — kể cả tuple C, vì "dẫn dắt trung nguyên xuống giếng" là hệ quả do chính các phẩm chất của bà Thảo gây ra, nên Target vẫn là người tạo ra hệ quả đó ("bà thảo"), không phải "trung nguyên" (Trung Nguyên ở đây là đối tượng bị tác động, không phải đối tượng bị đánh giá trực tiếp trong mệnh đề này).

→ Target = "bà thảo" cho cả 3 tuple.

Bước 5: Phân loại Polarity
Tuple A — "bộ óc sáng tạo cùng với sự lãnh đạo tài giỏi": từ ngữ "sáng tạo", "tài giỏi" mang tính khen ngợi năng lực → Positive.
Tuple B — "lòng tham hơn người và trí tuệ ngắn hạn": "lòng tham", "ngắn hạn" mang tính phê phán phẩm chất xấu → Negative.
Tuple C — "sẽ dẫn dắt trung nguyên xuống con mẹ nó giếng luôn": thành ngữ ẩn dụ chỉ kết cục tồi tệ/sụp đổ, kèm ngôn từ thô tục nhấn mạnh mức độ tiêu cực → Negative.

""",
            "output": """
{
  "opinions": [
    {
      "Source": [],
      "Target": ["bà thảo"],
      "Polar_expression": ["bộ óc sáng tạo cùng với sự lãnh đạo tài giỏi"],
      "Polarity": "Positive"
    },
    {
      "Source": [],
      "Target": ["bà thảo"],
      "Polar_expression": ["lòng tham hơn người và trí tuệ ngắn hạn"],
      "Polarity": "Negative"
    },
    {
      "Source": [],
      "Target": ["bà thảo"],
      "Polar_expression": ["sẽ dẫn dắt trung nguyên xuống con mẹ nó giếng luôn"],
      "Polarity": "Negative"
    }
  ]
}
"""
        },
        {
            "text": "Anh em với tập cận bình bảo anh dạy cách chống tham nhũng rất rễ làm và có dám làm không",
            "reasoning": """
Bước 1: Xác định số mệnh đề/ý kiến độc lập
Phân tích cấu trúc câu, ta thấy có 2 mệnh đề được nối bằng "và":

Mệnh đề 1: "Anh em với tập cận bình bảo anh dạy cách chống tham nhũng rất rễ làm" → một phát biểu/tuyên bố (claim) rằng việc dạy chống tham nhũng "rất dễ làm".
Mệnh đề 2: "có dám làm không" → một câu hỏi tu từ chất vấn lại liệu có dám thực hiện hay không.

Hai mệnh đề này có cực tính khác nhau (một mang tính khẳng định tích cực về độ dễ, một là câu hỏi nghi vấn trung tính) → tách thành 2 tuple riêng biệt.

Bước 2: Xác định Polar_expression cho từng mệnh đề
Tuple 1: "dạy cách chống tham nhũng rất rễ làm" (vị trí 32:68) — trích nguyên văn, giữ teencode "rễ" (không chuẩn hóa thành "dễ") theo quy tắc trích xuất nguyên văn (exact span).
Tuple 2: "có dám làm không" (vị trí 72:88) — văn bản gốc không có dấu ? ở cuối nên giữ nguyên toàn bộ cụm từ, không cần cắt bớt ký tự nào.

Bước 3: Xác định Source cho từng tuple
Tuple 1: Xét động từ "dạy" trong cụm "anh dạy cách chống tham nhũng..." — đại từ "anh" (xuất hiện lần thứ 2, sau "bảo", tại vị trí 28:31) đứng ngay trước động từ "dạy" và đóng vai trò chủ ngữ trực tiếp của hành động/biểu thức cảm xúc này. Đây là trường hợp đại từ xuất hiện tường minh, độc lập về cú pháp (không nằm trong cụm danh từ khác, không phải tân ngữ) → theo Mục A, được trích xuất làm Source = "anh" (28:31).
Tuple 2: Mệnh đề "có dám làm không" bị lược chủ ngữ hoàn toàn (không có "anh", "ai" hay đại từ nào xuất hiện tường minh ngay trong cụm này) → áp dụng quy tắc Pro-drop, Source = [].

Bước 4: Xác định Target cho từng tuple
Tuple 1: Đối tượng được nhắc tới làm chủ đề của lời thuật lại — "Anh em" (vị trí 0:6) — là người/nhóm được "Tập Cận Bình bảo" và bị gán cho hành động "dạy cách chống tham nhũng rất rễ làm". Đây là cụm danh từ độc lập, tường minh, đứng ở đầu câu, đóng vai trò là đối tượng chính của toàn bộ phát biểu (người bị đánh giá/nhắc đến trong câu chuyện) → Target = "Anh em" (0:6). Lưu ý: khác với Source ("anh" — chủ ngữ cú pháp trong mệnh đề nhúng "anh dạy..."), Target ở đây là thực thể được nói đến trong toàn cảnh câu (người được Tập Cận Bình bảo).
Tuple 2: "có dám làm không" là câu hỏi tu từ chung chung, không có thực thể cụ thể nào được nêu tên làm đối tượng đánh giá trong chính mệnh đề này → Target = [].

Bước 5: Phân loại Polarity
Tuple 1 — "dạy cách chống tham nhũng rất rễ làm": từ "rất rễ (dễ) làm" mang sắc thái khẳng định khả năng/sự dễ dàng, một đặc điểm được nêu ra theo hướng tích cực (dễ làm = có lợi, khả thi) → Positive.
Tuple 2 — "có dám làm không": đây là câu hỏi nghi vấn/tu từ, không mang sắc thái yêu/ghét rõ rệt, chỉ chất vấn về việc có thực hiện hay không → theo Mục D (câu hỏi tu từ, câu hỏi nghi vấn → Neutral) → Neutral.

""",
            "output": """
{
  "opinions": [
    {
      "Source": ["anh"],
      "Target": ["Anh em"],
      "Polar_expression": ["dạy cách chống tham nhũng rất rễ làm"],
      "Polarity": "Positive"
    },
    {
      "Source": [],
      "Target": [],
      "Polar_expression": ["có dám làm không"],
      "Polarity": "Neutral"
    }
  ]
}
"""
        },
        {
            "text": "bài viết hay quá !",
            "reasoning": """
Bước 1: Xác định số mệnh đề/ý kiến độc lập
Đây là một câu đơn, ngắn gọn, chỉ chứa một mệnh đề đánh giá duy nhất: nhận xét về "bài viết" là "hay quá". Không có sự phân tách thành nhiều khía cạnh hay nhiều cực tính khác nhau → chỉ tạo 1 tuple.

Bước 2: Xác định Polar_expression
Cụm từ thể hiện đánh giá là "hay quá" (vị trí 9:16). Theo quy tắc loại bỏ dấu câu cuối câu (Mục 1), dấu ! ở cuối văn bản không được đưa vào span của Polar_expression.
→ Polar_expression = "hay quá" (9:16), không lấy !.

Bước 3: Xác định Source
Toàn câu không có đại từ nhân xưng hay danh từ nào đóng vai trò chủ thể phát ngôn tường minh (không có "tôi thấy", "mình nghĩ"...). Đây là một nhận xét trực tiếp, chủ ngữ phát biểu hoàn toàn bị ẩn/lược bỏ.
→ Theo quy tắc Pro-drop (Mục A): Source = [].

Bước 4: Xác định Target
Cụm danh từ "bài viết" (vị trí 0:8) xuất hiện tường minh ở đầu câu, là đối tượng cụ thể đang bị đánh giá bởi Polar_expression "hay quá". Đây không phải trường hợp mệnh đề trọn vẹn không tách được Target — có một thực thể rõ ràng (bài viết) là chủ ngữ ngữ pháp được mô tả tính chất.
→ Target = "bài viết" (0:8).

Bước 5: Phân loại Polarity 
"hay quá": từ "hay" mang nghĩa khen ngợi, đánh giá tích cực về chất lượng nội dung, kết hợp "quá" nhấn mạnh mức độ khen → Positive.

""",
            "output": """
{
  "opinions": [
    {
      "Source": [],
      "Target": ["bài viết"],
      "Polar_expression": ["hay quá"],
      "Polarity": "Positive"
    }
  ]
}
"""
        },
        {
            "text": "t chốg mắt lên xem chúg mày mặn nồg đến bao giờ 🙃",
            "reasoning": """
Bước 1: Xác định số mệnh đề/ý kiến độc lập
Phân tách câu thành các phần:

"t chốg mắt lên xem" — hành động thách thức/khiêu khích do "t" thực hiện (chủ ngữ rõ ràng).
"chúg mày mặn nồg đến bao giờ" — câu hỏi mỉa mai, châm biếm hướng tới "chúg mày" về việc "mặn nồng" (tình cảm thắm thiết) sẽ kéo dài đến khi nào.

Hai cụm này có cấu trúc chủ ngữ khác nhau (một có Source tường minh, một không) và là hai hành vi/đánh giá riêng biệt → tách thành 2 tuple.
Lưu ý theo quy tắc loại bỏ ký tự cảm xúc mạng: emoji 🙃 ở cuối câu không được đưa vào bất kỳ span nào.

Bước 2: Xác định Polar_expression cho từng mệnh đề
Tuple 1: "chốg mắt lên xem" (2:18) — giữ nguyên teencode "chốg" (không chuẩn hóa thành "trừng"/"chong"), trích nguyên văn theo exact span.
Tuple 2: "mặn nồg đến bao giờ" (28:47) — giữ nguyên teencode "nồg" (không chuẩn hóa thành "nồng").

Bước 3: Xác định Source cho từng tuple
Tuple 1: "t" (0:1) đứng ở đầu câu, là chủ ngữ trực tiếp của động từ "chốg mắt lên xem" — đại từ xuất hiện tường minh, độc lập về cú pháp, không nằm trong cụm danh từ khác → theo Mục A, trích xuất làm Source = "t" (0:1).
Tuple 2: Trong mệnh đề "chúg mày mặn nồg đến bao giờ", "chúg mày" tuy đứng ở vị trí chủ ngữ ngữ pháp của tính từ "mặn nồng", nhưng đây là đối tượng bị mô tả/đánh giá trạng thái (bị châm biếm là "mặn nồng"), không phải người phát biểu ý kiến. Không có chủ thể nào khác đứng ra nhận định/phát ngôn tường minh trong mệnh đề này → Source = [].

Bước 4: Xác định Target cho từng tuple
Tuple 1: Hành động "chốg mắt lên xem" của "t" được hướng tới ai? Xét toàn câu, đối tượng bị thách thức/quan sát chính là "chúg mày" (19:27) — xuất hiện tường minh ngay sau đó, đóng vai trò là đối tượng nhận tác động của ánh nhìn thách thức này → Target = "chúg mày" (19:27).
Tuple 2: "chúg mày" (19:27) chính là chủ thể bị mô tả "mặn nồng" trong câu hỏi mỉa mai → Target = "chúg mày" (19:27), đầy đủ cụm danh từ chỉ người được nhắc đến.

Bước 5: Phân loại Polarity  
Tuple 1 — "chốg mắt lên xem": hành động mang tính đe dọa, thách thức, quan sát dò xét với ý đồ tiêu cực (chờ xem đối phương thất bại/lộ bản chất) → Negative.
Tuple 2 — "mặn nồg đến bao giờ": câu hỏi tu từ mang tính mỉa mai, châm biếm, ngầm ý nghi ngờ/chê bai tình cảm của đối tượng sẽ không bền lâu → Negative (không xếp Neutral vì có sắc thái châm biếm/khinh thường rõ rệt, khác với câu hỏi khách quan thuần túy).

""",
            "output": """
{
  "opinions": [
    {
      "Source": ["t"],
      "Target": ["chúg mày"],
      "Polar_expression": ["chốg mắt lên xem"],
      "Polarity": "Negative"
    },
    {
      "Source": [],
      "Target": ["chúg mày"],
      "Polar_expression": ["mặn nồg đến bao giờ"],
      "Polarity": "Negative"
    }
  ]
}
"""
        },
        {
            "text": "mình nói thật nhìn từ trên cao xuống đã thấy chóng mặt rồi , vãi",
            "reasoning": """
Bước 1: Xác định số mệnh đề/ý kiến độc lập
Câu gồm các phần:

"mình nói thật" — cụm mở đầu mang tính khẳng định tính chân thật của phát biểu (discourse marker), không tự thân là một đánh giá cảm xúc.
"nhìn từ trên cao xuống đã thấy chóng mặt rồi" — mệnh đề chính: mô tả nguyên nhân (nhìn từ trên cao xuống) và kết quả/trạng thái (thấy chóng mặt).
", vãi" — phần đứng sau dấu phẩy, tách rời khỏi mệnh đề chính, không có động từ/vị ngữ đi kèm để tạo thành một mệnh đề hoàn chỉnh và có thể đánh giá được độc lập.

→ Chỉ có 1 ý kiến/mệnh đề trọn vẹn đáng trích xuất là phần mô tả nguyên nhân–kết quả của việc nhìn xuống từ trên cao. Cụm ", vãi" đứng tách biệt sau dấu phẩy, không gắn liền cú pháp với một vị ngữ cụ thể trong cùng câu nên không tạo thành một biểu thức cảm xúc độc lập, có thể trích xuất được → không tách thành tuple riêng và cũng không được gộp vào Polar_expression đã có (vì Polar_expression phải là chuỗi liên tục).

Bước 2: Xác định Polar_expression
"đã thấy chóng mặt rồi" (37:58) — mô tả trạng thái cơ thể (cảm giác chóng mặt) là kết quả trực tiếp của hành động nhìn xuống từ trên cao. Đây là biểu thức trọn vẹn, trích nguyên văn, không kèm dấu câu/emoji.

Bước 3: Xác định Source
"mình" (0:4) đứng ở đầu câu, là chủ ngữ tường minh, độc lập về cú pháp. Mặc dù về mặt câu chữ "mình" trực tiếp đứng trước "nói thật", nhưng toàn câu là một chuỗi mệnh đề cùng chủ ngữ (đặc trưng lược chủ ngữ liên tiếp trong tiếng Việt): "mình (nói thật) [mình] nhìn... [mình] đã thấy chóng mặt..." — chủ ngữ "mình" được hiểu xuyên suốt cho cả hành động "thấy". Vì "mình" xuất hiện tường minh, không nằm trong cụm danh từ khác và không phải tân ngữ → trích xuất làm Source = "mình" (0:4).

Bước 4: Xác định Target
Xét nguyên nhân gây ra cảm giác "chóng mặt" — đó là hành động "nhìn từ trên cao xuống" (14:36). Đây là một hành động/sự việc cụ thể, tách biệt rõ ràng khỏi phần mô tả cảm giác kết quả ("đã thấy chóng mặt rồi"), nên được trích xuất làm Target (đối tượng/sự việc gây ra trạng thái), thay vì gộp chung toàn bộ câu vào một Polar_expression duy nhất.
→ Target = "nhìn từ trên cao xuống" (14:36).

Bước 5: Phân loại Polarity  
"đã thấy chóng mặt rồi" là một mô tả mang tính thực tế khách quan về phản ứng sinh lý khi nhìn xuống từ độ cao (hiện tượng chóng mặt do độ cao), không phải một lời khen/chê hay biểu lộ cảm xúc yêu/ghét rõ rệt đối với đối tượng nào. Theo Mục D, các phát biểu mô tả trạng thái thực tế khách quan thuộc nhóm Neutral.
→ Polarity = Neutral.
""",
            "output": """
{
  "opinions": [
    {
      "Source": ["mình"],
      "Target": ["nhìn từ trên cao xuống"],
      "Polar_expression": ["đã thấy chóng mặt rồi"],
      "Polarity": "Neutral"
    }
  ]
}
"""
        }
    ],  
    "opener_en": [
    {
        "text": """Yes , it really was a great experience and we visited various places but the most wonderful part of the trip was our stay at the Oberoi Udaivilas Luxury Hotel .""",
        "reasoning": """Step 1: The opening stretch confirms the trip went well and lists that various places were visited; the "but" then singles out one element of the trip as standing above the rest. The point the sentence is built to make is that superlative singling-out, and the earlier affirmation about the trip as a whole is the backdrop it is measured against. One opinion.

Step 2: The first-person plural appears only as the people who travelled and stayed somewhere, not as an announced holder of the judgment, which is stated as a fact about the trip. Source = [].

Step 3: What is elevated above everything else is the stay at a named hotel, and the hotel itself is the thing being praised — named in full as "the Oberoi Udaivilas Luxury Hotel". Shortening the name would leave the property unidentified. Target = ["the Oberoi Udaivilas Luxury Hotel"].

Step 4: The praise is delivered by "the most wonderful part of the trip". The superlative needs the comparison class attached to it, because the compliment consists precisely in ranking the hotel above every other element of the trip; "wonderful" alone would state approval but lose the ranking that is the sentence's actual claim.

Step 5: Being named the finest part of a trip already described as a great experience is unambiguous praise, with no irony or hedging present. Polarity = Positive.

Step 6: One tuple for the single ranking claim, with the lead-in treated as the backdrop rather than an extra opinion. Both spans were checked against the sentence and appear in it exactly as written.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["the Oberoi Udaivilas Luxury Hotel"],
                    "Polar_expression": ["the most wonderful part of the trip"],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """I can ât explain in words how grand this place looks .""",
        "reasoning": """Step 1: The sentence has a framing clause about being unable to put something into words, wrapped around one evaluation of how a place appears. The inability is a way of amplifying that evaluation, not a judgment of its own, so there is one opinion.

Step 2: The "I" is the subject of the framing clause about explaining, and the appraisal of the place is not separately attributed to a stated holder. Source = [].

Step 3: The thing appraised is "this place", which is how the sentence refers to the property being described. No shorter or more specific mention of it appears. Target = ["this place"].

Step 4: Two pieces carry the appraisal: "how grand", which states the quality and its degree, and "looks", which is what ties that quality to the appearance of the place rather than to something else about it. The Target sits between them, which is why the span comes in two parts.

Step 5: Grandeur described as beyond words is high praise of a property's appearance, with nothing to reverse the reading. Polarity = Positive.

Step 6: One tuple for one evaluation; the framing clause was not treated as separate. Every extracted piece, including the unusual character sequence left untouched in the contraction, was checked against the sentence and matches it.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["this place"],
                    "Polar_expression": ["how grand", "looks"],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """It is a unique blend of the old world royal charm and the modern luxuries .""",
        "reasoning": """Step 1: A single predication characterises something as a particular kind of combination. There are no further clauses, so one opinion.

Step 2: No speaker or holder is mentioned anywhere; the characterisation is asserted flatly. Source = [].

Step 3: The thing characterised is referred to only by the pronoun "It", which stands for the property under discussion and is the sentence's sole mention of what is evaluated. Target = ["It"].

Step 4: The appreciative content runs the length of the predicate: "unique" rates the combination as one of a kind, and the two coordinated halves — the old-world charm and the modern luxuries — are what make the combination worth remarking on. Cutting the coordination would leave a bare claim of uniqueness with nothing said about what is being blended, and the copula and article are not part of the assessment.

Step 5: Both ingredients named are desirable ones, and calling their combination unique presents the mix as a distinction rather than an oddity. Polarity = Positive.

Step 6: One opinion, one tuple, all four fields filled. The long evaluative span and the pronoun Target are exact substrings of the sentence, and the span cannot be trimmed without losing part of what is praised.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["It"],
                    "Polar_expression": [
                        "unique blend of the old world royal charm and the modern luxuries"
                    ],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """I âm definitely going there again whenever I get a chance .""",
        "reasoning": """Step 1: The sentence states an intention to return to a place. A stated intention to go back is itself how the writer expresses approval of the place, and the trailing condition about getting a chance only qualifies when it will happen. One opinion.

Step 2: The writer names themselves as the one committing to return, with "I" at the head of the sentence. That pronoun is an explicit mention of the holder, not something reconstructed. Source = ["I"].

Step 3: The place is referred to only by "there", the sentence's sole mention of the destination. Target = ["there"].

Step 4: The endorsement rests on "definitely going", where the adverb turns a plan into a firm commitment, together with "again", which is the piece that makes it a return and therefore an endorsement of a previous visit. The Target sits between the two, leaving the span in two parts, and the concluding condition is left out because it only sets the timing.

Step 5: Firmly committing to go back at the first opportunity conveys satisfaction with the earlier stay. Polarity = Positive.

Step 6: One tuple for the single intention, with the timing condition excluded. All spans were verified against the sentence and appear in it exactly, the odd characters in the contraction included.""",
        "output": {
            "opinions": [
                {
                    "Source": ["I"],
                    "Target": ["there"],
                    "Polar_expression": ["definitely going", "again"],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """Bit pricey and but away from center""",
        "reasoning": """Step 1: This is a verbless review fragment listing two separate drawbacks: one about cost, one about location. They concern different properties of the place and neither explains the other, so there are two independent opinions. The doubled connective is just clumsy joining.

Step 2: Nothing in the fragment names or refers to a person; it is a bare list of drawbacks. Source = [] for both.

Step 3: The fragment never names what is expensive or what is far out — the subject is left entirely unstated, and the only noun present, "center", is the reference point the distance is measured from rather than the thing being judged. Target = [] for both.

Step 4: The first drawback is "Bit pricey", where the hedge softens the complaint but is part of how strongly it is made. The second is "away from center", which needs all three words: the distance claim only becomes a complaint once it says what the place is away from.

Step 5: Costing more than one would like is a drawback, and in a review of somewhere to stay, being out of the centre is raised as an inconvenience. Polarity = Negative for both.

Step 6: Two tuples matching the two drawbacks identified at the start, with the connective left out of both. Each Polar_expression is an exact substring of the fragment, and both Target fields are correctly empty since no evaluated entity is named.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": [],
                    "Polar_expression": ["Bit pricey"],
                    "Polarity": "Negative",
                },
                {
                    "Source": [],
                    "Target": [],
                    "Polar_expression": ["away from center"],
                    "Polarity": "Negative",
                },
            ]
        },
    },
    {
        "text": """You need to count 20 Â· 30 minutes walking time to get to Dubling city center from this hotel .""",
        "reasoning": """Step 1: The sentence makes one point: how much walking has to be reckoned with to reach the city centre. It is framed as advice to the reader but amounts to a single complaint about distance.

Step 2: The "You" addressed is the reader being advised, not a holder claiming the view, and no other person is mentioned. Source = [].

Step 3: Nothing here is named as the entity being judged — what the sentence complains about is the walk, and the hotel is mentioned only as the point the walk starts from, while the city centre is where it ends. Neither is presented as the thing evaluated. Target = [].

Step 4: The complaint is made by "20 Â· 30 minutes walking" together with "to Dubling city center". A stretch of walking time on its own says nothing against the place; it becomes a drawback only once it is stated what all that walking buys you, namely reaching the centre. The words in between, about time and getting there, are grammatical connective material, so the span comes in two parts.

Step 5: Half an hour on foot to reach the city centre is presented here as something the reader must reckon with, i.e. as an inconvenience of staying at this hotel. Polarity = Negative.

Step 6: One tuple for the single point about distance, with both fragments needed and verified against the sentence exactly as written, the unusual character between the two numbers left untouched. Source and Target are empty because neither a holder nor an evaluated entity is named.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": [],
                    "Polar_expression": [
                        "20 Â· 30 minutes walking",
                        "to Dubling city center",
                    ],
                    "Polarity": "Negative",
                }
            ]
        },
    },
    {
        "text": """There is no sauna nor swimming pool .""",
        "reasoning": """Step 1: One existential statement denies two amenities. Each missing facility is a separate shortcoming of the place — a guest who wanted one and not the other is affected differently — so this is two opinions sharing a single statement of absence.

Step 2: No holder is mentioned; the absence is simply reported. Source = [] for both.

Step 3: The two things whose absence is noted are "sauna" and "swimming pool", each named exactly once. Each becomes the Target of its own tuple; the coordinator between them belongs to neither.

Step 4: What conveys the shortcoming in both cases is "There is no" — the statement of non-existence that governs both nouns. Since one statement covers them both, the same span serves as the Polar_expression of each tuple.

Step 5: A sauna and a pool are facilities guests look for, so being told the place has neither is a mark against it; the absence of something desirable is what makes this negative rather than the negator itself. Polarity = Negative for both.

Step 6: Two tuples, one per missing facility, as counted at the start. The shared span and both Targets are exact substrings of the sentence, and none of them can be shortened further.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["sauna"],
                    "Polar_expression": ["There is no"],
                    "Polarity": "Negative",
                },
                {
                    "Source": [],
                    "Target": ["swimming pool"],
                    "Polar_expression": ["There is no"],
                    "Polarity": "Negative",
                },
            ]
        },
    },
    {
        "text": """Flat TV with discovery but no HBO .""",
        "reasoning": """Step 1: The fragment reports what the television offers and what it lacks. The "but" marks a shift from credit to complaint, so there is a favourable point and a shortcoming; the shortcoming is registered twice over, once by naming the channel that is missing and once by the bare denial itself, since each carries a distinct part of the grievance — which channel is absent, and the fact of the absence. Three opinions.

Step 2: The fragment is written in note form with nobody mentioned. Source = [] for all three.

Step 3: Everything said here is said about the television, named as "Flat TV" at the front of the fragment. The channels are what it does or does not carry rather than the thing being judged, so all three tuples share Target = ["Flat TV"].

Step 4: The credit is "with discovery" — the preposition matters, since it is what states that the channel is included. For the complaint, "HBO" names the channel whose absence is regretted, and "no" states that it is not there; neither word can stand in for the other, so each is recorded as its own evaluative span.

Step 5: Having a named channel available is a point in the television's favour, so the first is Positive. Missing a channel the writer evidently wanted counts against it, so both parts of the complaint are Negative.

Step 6: Three tuples, matching the credit and the two components of the shortcoming identified at the start. All spans, including the lower-case channel name as it is written here, appear in the fragment exactly and are as short as they can be.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["Flat TV"],
                    "Polar_expression": ["with discovery"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["Flat TV"],
                    "Polar_expression": ["HBO"],
                    "Polarity": "Negative",
                },
                {
                    "Source": [],
                    "Target": ["Flat TV"],
                    "Polar_expression": ["no"],
                    "Polarity": "Negative",
                },
            ]
        },
    },
    {
        "text": """In my room mattress was ok and it was quiet but I had room with view on wall : ) Most of colleagues were complaining for the mattresses being old and some for noises from outside ( train in particular ) .""",
        "reasoning": """Step 1: Several separate judgments are packed in here. From the writer's own room: the mattress was acceptable, the room was quiet, and the outlook was onto a wall. Then a reported strand: colleagues complained about the mattresses, which is registered in more than one piece because the complaint has both an act of complaining and a stated fault. A further reported grievance about outside noise also appears. Each of these concerns a different aspect and none merely restates another.

Step 2: The judgments about the writer's own room are stated without the writer claiming them, except for the outlook, which is introduced with "I had" — there the writer is named as the one reporting the experience, so Source = ["I"] for that tuple and [] for the mattress and the quiet. The reported grievances are attributed to "Most of colleagues", the phrase the sentence uses to name those complaining.

Step 3: The mattress in the writer's own room is named as "mattress"; the quiet is asserted of "it", the sentence's only reference to the room in that clause; the outlook belongs to "room". In the reported strand, what the colleagues complain about is "the mattresses" — the plural form, distinct from the writer's single mattress.

Step 4: "ok" is the whole of the mattress verdict, a bare acceptability rating with nothing added. "quiet" carries the second. "with view on wall" is the third and needs all four words, since the complaint lies in what the view is of, not in having a view. For the colleagues' strand, "complaining" states that a grievance was voiced and "being old" states the fault found, each a different part of the same report. The remaining span, "noises", names the disturbance raised in the final clause.

Step 5: An acceptable mattress and a quiet room are stated as satisfactory, so both are Positive. Looking out onto a wall is offered as a disappointment against the two preceding pluses, marked by the "but", so it is Negative. Voicing complaints, calling mattresses old, and reporting noise are all unfavourable, so those are Negative.

Step 6: All the strands identified at the start are represented and none was dropped, and every span was verified to occur in the sentence exactly, including the emoticon-bearing stretch which contributes no extracted span. One point does not hold up on rechecking: the span "noises" belongs to the clause about disturbance from outside, attributed there to "some" and concerning the train, so it is not an evaluation of the mattresses and not something "Most of colleagues" is said to have raised. Pairing it with the mattresses as its Target cannot be supported by any reading of this sentence.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["mattress"],
                    "Polar_expression": ["ok"],
                    "Polarity": "Positive",
                },
                {
                    "Source": ["Most of colleagues"],
                    "Target": ["the mattresses"],
                    "Polar_expression": ["noises"],
                    "Polarity": "Negative",
                },
                {
                    "Source": ["Most of colleagues"],
                    "Target": ["the mattresses"],
                    "Polar_expression": ["complaining"],
                    "Polarity": "Negative",
                },
                {
                    "Source": ["Most of colleagues"],
                    "Target": ["the mattresses"],
                    "Polar_expression": ["being old"],
                    "Polarity": "Negative",
                },
                {
                    "Source": [],
                    "Target": ["it"],
                    "Polar_expression": ["quiet"],
                    "Polarity": "Positive",
                },
                {
                    "Source": ["I"],
                    "Target": ["room"],
                    "Polar_expression": ["with view on wall"],
                    "Polarity": "Negative",
                },
            ]
        },
    },
    {
        "text": """Service on the other hand was very good and the stuff is quite international : )""",
        "reasoning": """Step 1: Two separate compliments are coordinated: one about the service, one about the staff's international character. They concern different things and neither explains the other, so two opinions. The contrastive opener positions this against something said earlier and adds no judgment of its own.

Step 2: Neither compliment is attributed to anyone; both are stated as plain assessments. Source = [] for both.

Step 3: The staff, written here as "the stuff", is what the second compliment is about, and "Service" is what the first is about. Each is named once and neither can be shortened.

Step 4: The second compliment is "quite international", where the degree word is part of how the claim is pitched and the adjective is the quality credited. The first is "very good", where the intensifier raises a plain approval into a strong one; the copula in each case is grammatical support only, and the closing emoticon reinforces the tone without being needed for either assessment.

Step 5: In a review of a place to stay, an internationally mixed staff is offered as an asset, so that is Positive; service rated as very good is straightforward praise, likewise Positive.

Step 6: Two tuples for the two compliments counted at the start, with the contrastive opener and the emoticon left out. Every span occurs in the sentence exactly as written, including the non-standard spelling of the staff, which was kept as it stands.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["the stuff"],
                    "Polar_expression": ["quite international"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["Service"],
                    "Polar_expression": ["very good"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
],
    "opener_es": [
    {
        "text": """Encanto real""",
        "reasoning": """Step 1: Se trata de un título de reseña de dos palabras, sin verbo: un sustantivo valorativo reforzado por un adjetivo. Hay una sola opinión.

Step 2: No aparece ninguna palabra que señale quién emite la valoración. Source = [].

Step 3: El fragmento no nombra aquello que resulta encantador: no hay sustantivo que designe el establecimiento ni ningún otro elemento evaluado, solo la cualidad atribuida. Target = [].

Step 4: La valoración la constituyen las dos palabras juntas: "Encanto" nombra la cualidad y "real" afirma que no es aparente sino auténtica, de modo que quitar el adjetivo debilitaría lo que se afirma. Polar_expression = ["Encanto real"].

Step 5: Atribuir encanto verdadero a un alojamiento es un elogio directo, sin ironía ni atenuación. Polarity = Positive.

Step 6: Una opinión y una tupla, con los cuatro campos. El fragmento extraído coincide exactamente con el texto y los campos de fuente y objeto quedan vacíos porque ninguno de los dos se menciona.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": [],
                    "Polar_expression": ["Encanto real"],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """volveria si tuviera la ocasion me he sentido como una princesa en un cuento maravillosooooo.las vistas magicas el personal superior en calidad y la comida sin palabras como que he cojido un par de kilos despues de mi estancia .""",
        "reasoning": """Step 1: El texto va sin puntuación que separe las partes y enumera varias valoraciones distintas: la sensación de la propia estancia, las vistas, el personal y la comida. La apertura sobre volver si se tuviera ocasión plantea una hipótesis sobre el futuro sin calificar nada, y el cierre sobre los kilos ganados explica de qué modo se comió bien. Se identifican cuatro opiniones.

Step 2: La única marca explícita de quién valora es el pronombre "me", que introduce la experiencia vivida en primera persona; corresponde solo a la tupla de la sensación personal. En las valoraciones de vistas, personal y comida no hay ninguna palabra que designe a quien las emite, así que ahí Source = [].

Step 3: Los elementos valorados son "vistas", "la comida" y "el personal", cada uno con la forma exacta con que aparece. En la tupla de la sensación personal no hay entidad evaluada: lo que se describe es cómo se sintió quien escribe, y la princesa del cuento es la comparación, no el objeto valorado. Target = [] en ese caso.

Step 4: "magicas" basta para las vistas y "superior en calidad" para el personal, donde el complemento precisa en qué es superior. Para la comida, "sin palabras" es la fórmula entera que expresa el elogio: ni la preposición ni el sustantivo por separado dicen nada. Para la sensación personal, el fragmento va desde "he sentido" hasta el adjetivo alargado con vocales repetidas, que es donde está la intensidad del elogio; ese adjetivo aparece escrito pegado sin espacio a la palabra siguiente, y el fragmento se conserva tal como está en el texto.

Step 5: Sentirse como una princesa, unas vistas mágicas, un personal superior en calidad y una comida indescriptible son todos elogios; el comentario sobre los kilos confirma que la comida se valora bien. Polarity = Positive en las cuatro.

Step 6: Cuatro tuplas para las cuatro valoraciones contadas al principio, sin convertir la hipótesis inicial ni el comentario final en opiniones adicionales. Todos los fragmentos se han comprobado contra el texto y coinciden con él tal cual, faltas de acentuación y letras repetidas incluidas.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["vistas"],
                    "Polar_expression": ["magicas"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["la comida"],
                    "Polar_expression": ["sin palabras"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["el personal"],
                    "Polar_expression": ["superior en calidad"],
                    "Polarity": "Positive",
                },
                {
                    "Source": ["me"],
                    "Target": [],
                    "Polar_expression": [
                        "he sentido como una princesa en un cuento maravillosooooo.las"
                    ],
                    "Polarity": "Positive",
                },
            ]
        },
    },
    {
        "text": """Muy contentos""",
        "reasoning": """Step 1: Es un titular de dos palabras que expresa el estado de ánimo de quienes escriben. Hay una sola opinión.

Step 2: El plural del adjetivo indica que quienes valoran son varios, pero no hay ningún pronombre ni sustantivo que los nombre como palabra aparte. Source = [].

Step 3: El fragmento no menciona con qué están contentos: no hay ninguna entidad nombrada. Target = [].

Step 4: La expresión valorativa es "Muy contentos" entera: el adverbio gradúa la satisfacción y forma parte de la fuerza con que se afirma. Polar_expression = ["Muy contentos"].

Step 5: Declararse muy contento tras una estancia expresa satisfacción sin matices que la inviertan. Polarity = Positive.

Step 6: Una tupla para el único enunciado valorativo, con los cuatro campos presentes. El fragmento es subcadena exacta del texto y no se puede acortar sin perder el grado.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": [],
                    "Polar_expression": ["Muy contentos"],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """Es un camping encantador , muy natural y autentico .""",
        "reasoning": """Step 1: La frase atribuye tres cualidades al mismo establecimiento: el encanto, el carácter natural y la autenticidad. Son propiedades distintas — un camping puede ser encantador sin ser natural — de modo que hay tres opiniones sobre un mismo objeto.

Step 2: Ninguna palabra de la frase identifica a quien la afirma. Source = [] en las tres.

Step 3: El objeto valorado es "un camping", tal como lo nombra la frase; el artículo indeterminado forma parte de la mención y no hay otra manera más corta de designarlo. Sirve de Target en las tres tuplas.

Step 4: Cada cualidad se extrae por separado: "autentico", "encantador" y "muy natural", esta última con el adverbio, que gradúa hasta qué punto se le reconoce ese carácter. La cópula y la conjunción quedan fuera porque solo enlazan.

Step 5: En una reseña de camping, el encanto, el entorno natural y la autenticidad se presentan como atractivos del sitio. Polarity = Positive en las tres.

Step 6: Tres tuplas para las tres cualidades contadas al inicio, sin fundirlas en una sola. Todos los fragmentos coinciden exactamente con la frase, incluida la falta de tilde en el último adjetivo.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["un camping"],
                    "Polar_expression": ["autentico"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["un camping"],
                    "Polar_expression": ["encantador"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["un camping"],
                    "Polar_expression": ["muy natural"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
    {
        "text": """No tiene animacion , lo cual se agradece .""",
        "reasoning": """Step 1: La frase constata la ausencia de animación y, acto seguido, valora esa ausencia. Son dos cosas distintas: la constatación referida a la animación y el agradecimiento referido a la situación descrita, así que hay dos opiniones.

Step 2: El verbo de agradecimiento va en construcción impersonal y no hay pronombre ni sustantivo que nombre a quien agradece. Source = [] en ambas.

Step 3: La primera opinión versa sobre "animacion", el servicio cuya ausencia se señala. La segunda versa sobre "lo cual", que es la manera en que la frase recoge lo que se acaba de decir y lo convierte en objeto de la valoración siguiente; es una mención explícita de aquello que se agradece, no una reconstrucción.

Step 4: La primera se expresa con "No tiene", la construcción que enuncia la falta. La segunda, con "se agradece", el verbo que aporta la apreciación; el relativo pertenece al objeto valorado y no se incluye aquí.

Step 5: La segunda mitad de la frase dice expresamente que esa falta se agradece, de modo que quien escribe presenta la ausencia de animación como una ventaja: en este contexto, no tener animación es lo deseable, y por eso la constatación de la falta también es favorable. Polarity = Positive en las dos.

Step 6: Dos tuplas para las dos operaciones identificadas al principio, con los cuatro campos cada una. Los fragmentos son subcadenas exactas de la frase y ninguno puede recortarse sin perder su sentido.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["lo cual"],
                    "Polar_expression": ["se agradece"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["animacion"],
                    "Polar_expression": ["No tiene"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
    {
        "text": """Tiene todo lo necesario y sobretodo , lo que mas nos interesa : accesos excelentes al centro de Calella de Palafrugell y a la playa a pie .""",
        "reasoning": """Step 1: La frase afirma primero que el sitio dispone de lo necesario y después detalla lo que más interesa: los accesos al centro y el acceso a la playa. La calidad de los accesos al centro y el hecho de poder ir andando a la playa son ventajas distintas, referidas a destinos distintos, así que se cuentan tres opiniones.

Step 2: El pronombre de la fórmula "lo que mas nos interesa" señala a quienes escriben, pero pertenece al anuncio de lo que viene a continuación y no a ninguna de las tres valoraciones, que se enuncian como hechos del establecimiento. Source = [] en las tres.

Step 3: La primera valoración recae en "todo lo necesario", el conjunto de dotaciones de que dispone. La segunda, en "la playa", el destino al que se puede llegar. La tercera recae en los accesos al centro, y la mención se parte en dos: "accesos" y "al centro de Calella de Palafrugell", porque el adjetivo que los califica se cuela entre ambos; sin el complemento no sabríamos a dónde llevan esos accesos, que es justamente lo que interesa al viajero.

Step 4: "Tiene" es lo que expresa la disponibilidad de lo necesario y basta por sí solo. "excelentes" es el adjetivo que valora los accesos. Para la playa, la valoración es "a pie": la ventaja consiste precisamente en que se llega andando, y ninguna otra palabra de la frase expresa eso.

Step 5: Disponer de todo lo necesario, tener accesos excelentes y poder llegar a la playa caminando se presentan como puntos a favor del sitio. Polarity = Positive en las tres.

Step 6: Tres tuplas para las tres ventajas contadas al principio; la fórmula que introduce la enumeración no se ha convertido en opinión aparte. Todos los fragmentos, incluidos los dos del objeto partido, se han verificado contra la frase y aparecen en ella tal cual.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["todo lo necesario"],
                    "Polar_expression": ["Tiene"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["la playa"],
                    "Polar_expression": ["a pie"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["accesos", "al centro de Calella de Palafrugell"],
                    "Polar_expression": ["excelentes"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
    {
        "text": """Si que hay que caminar de cuesta arriba , pero es muy poquito rato y la cuesta no es nada pronunciada , de hecho , Calella es una colina , asi que no se porque se quejan algunos !""",
        "reasoning": """Step 1: La frase concede que hay que subir una cuesta y a continuación desmonta esa objeción. Dentro de ese desmontaje, lo que constituye una valoración de un elemento del entorno es la afirmación sobre la inclinación de la cuesta; la duración del trayecto se cuantifica, la mención de que Calella es una colina aporta un dato geográfico, y el cierre exclamativo se dirige a otros reseñadores en lugar de calificar el sitio. Se identifica una opinión.

Step 2: La frase no nombra a quien la escribe; el verbo final se refiere a terceros que se quejan y no a quien valora. Source = [].

Step 3: El elemento valorado es "la cuesta", que es lo que se dice que no tiene apenas inclinación. Es la mención más ajustada y no admite recorte. Target = ["la cuesta"].

Step 4: La valoración es "no es nada pronunciada" completa: la negación reforzada es lo que niega por entero la pendiente, y quedarse solo con el adjetivo invertiría el sentido de lo que se afirma.

Step 5: Aquí la ausencia de pendiente es lo deseable, porque toda la frase va encaminada a mostrar que la subida no es un problema; negar que la cuesta sea pronunciada juega, por tanto, en favor del sitio. Polarity = Positive.

Step 6: Una tupla para la única valoración de un elemento del entorno, sin convertir la concesión inicial ni el reproche final a otros en opiniones adicionales. Los dos fragmentos extraídos son subcadenas exactas de la frase.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["la cuesta"],
                    "Polar_expression": ["no es nada pronunciada"],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """Nosotros estuvimos muy bien pues todo esta limpio y muy tranquilo .""",
        "reasoning": """Step 1: La frase declara que la estancia fue buena y añade dos razones que la sustentan: la limpieza y la tranquilidad. La valoración general y las dos cualidades concretas son afirmaciones distintas sobre el mismo conjunto, de modo que se cuentan tres opiniones.

Step 2: Quienes valoran se nombran expresamente al comienzo con el pronombre "Nosotros", que encabeza la frase y sostiene tanto la declaración general como las razones que la justifican. Source = ["Nosotros"] en las tres.

Step 3: Las tres afirmaciones se refieren al conjunto del sitio, designado en la frase como "todo": es de ese conjunto de lo que se dice que está limpio y tranquilo, y es también aquello con lo que la estancia resultó buena. Target = ["todo"] en las tres.

Step 4: La valoración general es "muy bien", con el adverbio que la gradúa. Las dos concretas son "limpio" y "muy tranquilo", cada una con lo que aporta: en la segunda el intensificador forma parte del elogio. La conjunción causal solo articula la relación entre unas y otras.

Step 5: Estar muy bien, la limpieza y la tranquilidad son las tres cosas que quien escribe destaca a favor del sitio. Polarity = Positive en las tres.

Step 6: Tres tuplas para las tres afirmaciones contadas al inicio, sin que la relación causal entre ellas haya generado ninguna opinión de más. Todos los fragmentos coinciden exactamente con la frase.""",
        "output": {
            "opinions": [
                {
                    "Source": ["Nosotros"],
                    "Target": ["todo"],
                    "Polar_expression": ["muy bien"],
                    "Polarity": "Positive",
                },
                {
                    "Source": ["Nosotros"],
                    "Target": ["todo"],
                    "Polar_expression": ["limpio"],
                    "Polarity": "Positive",
                },
                {
                    "Source": ["Nosotros"],
                    "Target": ["todo"],
                    "Polar_expression": ["muy tranquilo"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
    {
        "text": """Los menus del bar Â· restaurante son muy buenos y estan muy bien de precio .""",
        "reasoning": """Step 1: La frase valora los menús en dos aspectos independientes: su calidad y su precio. Un menú puede ser bueno y caro, de modo que se trata de dos opiniones sobre el mismo objeto.

Step 2: No hay ninguna palabra que identifique a quien valora. Source = [] en ambas.

Step 3: El objeto valorado son los menús del establecimiento, designados como "Los menus del bar Â· restaurante". El complemento hace falta para saber de qué menús se habla, así que la mención se mantiene entera, con el símbolo que separa los dos nombres tal como aparece. Sirve de Target en las dos tuplas.

Step 4: La primera valoración es "muy buenos", que califica la calidad. La segunda es "muy bien de precio": la locución completa es la que dice que el precio es acertado, y recortarla dejaría un elogio genérico sin el aspecto al que se refiere.

Step 5: Que los menús sean muy buenos y además tengan buen precio son dos elogios claros del sitio. Polarity = Positive en las dos.

Step 6: Dos tuplas para los dos aspectos valorados, con los cuatro campos cada una. Los fragmentos se han comprobado contra la frase y aparecen en ella exactamente, con la grafía y el símbolo intermedio sin modificar.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["Los menus del bar Â· restaurante"],
                    "Polar_expression": ["muy bien de precio"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["Los menus del bar Â· restaurante"],
                    "Polar_expression": ["muy buenos"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
    {
        "text": """Correcto del todo , genial si viajas a ver el barÃ§a""",
        "reasoning": """Step 1: El fragmento contiene dos valoraciones: una general sobre el alojamiento y otra que lo recomienda para un propósito concreto, ir a ver al equipo. La segunda no repite la primera, porque acota el elogio a un tipo de viaje, así que hay dos opiniones.

Step 2: El fragmento se dirige al lector en segunda persona y no nombra a quien valora. Source = [] en ambas.

Step 3: La valoración general no menciona qué es lo correcto: no hay sustantivo que designe el establecimiento, así que Target = []. La segunda sí tiene objeto: lo que resulta genial es el viaje descrito como "ver el barÃ§a", que es el propósito para el cual se recomienda.

Step 4: La primera valoración es "Correcto del todo", donde el complemento es lo que lleva la corrección hasta el grado pleno. La segunda es "genial", el adjetivo que expresa el elogio; la condicional que sigue introduce el objeto valorado y no la valoración.

Step 5: Decir que algo está del todo correcto es aprobarlo, y calificar de genial la opción para ese viaje es recomendarlo con entusiasmo. Polarity = Positive en las dos.

Step 6: Dos tuplas para las dos valoraciones detectadas, con los cuatro campos. Los fragmentos son subcadenas exactas del texto, con el nombre del equipo en minúscula y la grafía tal como consta.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": [],
                    "Polar_expression": ["Correcto del todo"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["ver el barÃ§a"],
                    "Polar_expression": ["genial"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
],
    "norec": [
    {
        "text": """Eller bedrar skinnet ?""",
        "reasoning": """Step 1: Setningen er et kort, retorisk spørsmål som setter et tidligere inntrykk i tvil. Det er én vurderende ytring her, ikke flere.

Step 2: Ingen person eller instans nevnes som den som stiller spørsmålet. Source = [].

Step 3: Ingenting i setningen navngir det som vurderes — "skinnet" er selve inntrykket som eventuelt bedrar, altså en del av tvilen som uttrykkes, og ikke produktet den gjelder. Target = [].

Step 4: Uttrykket er "bedrar skinnet ?" med spørsmålstegnet, siden det nettopp er spørsmålsformen som gjør dette til en antydning om at inntrykket ikke holder. Uten tegnet ville det stå som en påstand med et annet innhold.

Step 5: Å spørre om skinnet bedrar er å foreslå at det gode førsteinntrykket ikke svarer til virkeligheten, altså et forbehold framfor ros. Polarity = Negative.

Step 6: Én tuppel for det ene retoriske spørsmålet, med alle fire feltene utfylt. Fragmentet er kontrollert mot setningen og stemmer tegn for tegn, mellomrommet foran spørsmålstegnet inkludert.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": [],
                    "Polar_expression": ["bedrar skinnet ?"],
                    "Polarity": "Negative",
                }
            ]
        },
    },
    {
        "text": """Baksiden er sort , blank og skinnende , med et deksel som skjuler kontakter og kabler .""",
        "reasoning": """Step 1: Setningen beskriver baksiden med tre egenskaper i en oppramsing og legger i tillegg til et deksel som skjuler tilkoblingene. Hver av de tre egenskapene sier noe eget om utseendet, og dekselet omtales både som noe baksiden har og som noe som gjør en konkret nytte. Det gir flere selvstendige vurderinger av det samme partiet.

Step 2: Ingen person står bak beskrivelsen i teksten; alt slås fast som egenskaper ved produktet. Source = [] i alle tuplene.

Step 3: Fire av vurderingene gjelder "Baksiden", som er det setningen beskriver. Den siste gjelder "deksel" selv, altså den delen som utfører skjulingen. Begge nevnes bare én gang hver og kan ikke kortes ned.

Step 4: De tre egenskapene står som "sort", "blank" og "skinnende", hver for seg, siden kommaene og konjunksjonen bare binder oppramsingen sammen. Til baksiden hører også "med et deksel som skjuler kontakter og kabler": preposisjonen er det som knytter dekselet til baksiden som en del av den. Om dekselet alene gjelder "skjuler kontakter og kabler" — det som sier hva det gjør.

Step 5: I en omtale av designet på et skjermkabinett regnes en sort, blank og skinnende bakside som pent utført, og et deksel som skjuler kontakter og kabler er nettopp det man ønsker seg, siden det holder rotet borte fra synet. Polarity = Positive i alle fem.

Step 6: Fem tupler for de fem vurderingene som ble skilt ut innledningsvis, uten at noen av dem er slått sammen eller utelatt. Alle fragmentene er kontrollert mot setningen og finnes der ordrett.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["Baksiden"],
                    "Polar_expression": [
                        "med et deksel som skjuler kontakter og kabler"
                    ],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["Baksiden"],
                    "Polar_expression": ["sort"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["Baksiden"],
                    "Polar_expression": ["blank"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["Baksiden"],
                    "Polar_expression": ["skinnende"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["deksel"],
                    "Polar_expression": ["skjuler kontakter og kabler"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
    {
        "text": """Skjermen er helt klart visuelt utfordrende , noen vil synes den er overharry , andre vil bli begeistret .""",
        "reasoning": """Step 1: Setningen slår først fast en egenskap ved skjermens utseende, og deler så publikum i to: noen vil reagere avvisende, andre entusiastisk. Det gir tre selvstendige vurderinger — én fra den som skriver, og to som tilskrives ulike grupper.

Step 2: Den første vurderingen står uten noen som gjør den til sin. De to andre har hver sin holder navngitt i teksten: "noen" og "andre", som er de eneste ordene som peker ut hvem reaksjonene tilhører.

Step 3: Alle tre gjelder "Skjermen", som er det setningen handler om; pronomenet senere i setningen viser bare tilbake til den og sier ikke noe nytt om hva som vurderes. Target = ["Skjermen"] i alle tre.

Step 4: Den første vurderingen er "visuelt utfordrende", der adverbet presiserer at det er det visuelle som utfordrer; forsterkeren foran sier bare hvor sikkert det slås fast og holdes utenfor. De to andre står som "vil synes den er overharry" og "vil bli begeistret", der modalverbet hører med fordi det er selve forutsigelsen av reaksjonen som utgjør vurderingen.

Step 5: Formuleringen "visuelt utfordrende" framheves her som noe ved designet som tør å skille seg ut, og setningen fortsetter med at en del vil bli begeistret — den umiddelbare karakteristikken av skjermen faller altså ut til dens fordel. Å oppfatte den som overharry er derimot en avvisende reaksjon, og begeistring er en positiv. Polarity = Positive, Negative, Positive.

Step 6: Tre tupler for de tre vurderingene som ble telt opp, med to ulike holdere holdt fra hverandre. Alle fragmentene er ordrette utsnitt av setningen og kan ikke kortes ned uten å miste innhold.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["Skjermen"],
                    "Polar_expression": ["visuelt utfordrende"],
                    "Polarity": "Positive",
                },
                {
                    "Source": ["noen"],
                    "Target": ["Skjermen"],
                    "Polar_expression": ["vil synes den er overharry"],
                    "Polarity": "Negative",
                },
                {
                    "Source": ["andre"],
                    "Target": ["Skjermen"],
                    "Polar_expression": ["vil bli begeistret"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
    {
        "text": """God betjening""",
        "reasoning": """Step 1: Dette er en mellomtittel på to ord: et adjektiv som karakteriserer én egenskap ved produktet. Én vurdering.

Step 2: Ingen som vurderer er nevnt. Source = [].

Step 3: Det som vurderes, er hvordan produktet betjenes, uttrykt som "betjening". Ordet nevnes bare én gang og lar seg ikke korte ned. Target = ["betjening"].

Step 4: Hele vurderingen ligger i adjektivet "God", som er det som tilskriver kvaliteten. Ingenting annet i overskriften bidrar.

Step 5: God betjening er en klar fordel ved et produkt som skal styres av brukeren. Polarity = Positive.

Step 6: Én tuppel med alle fire feltene. Begge fragmentene er kontrollert mot teksten og stemmer, stor forbokstav i adjektivet inkludert.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["betjening"],
                    "Polar_expression": ["God"],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """Uansett hva man mÃ¥tte mene om utseendet , sÃ¥ er knappene plassert pÃ¥ en utmerket mÃ¥te , og hele oppsettet gir god kontroll - langt mer behagelig i bruk enn de vanlige tryknappene som vi finner pÃ¥ de aller fleste skjermer .""",
        "reasoning": """Step 1: Innledningen setter uenigheten om utseendet til side uten selv å vurdere noe. Deretter kommer tre påstander: at knappene er godt plassert, at oppsettet gir god kontroll, og at det er langt behageligere i bruk enn de vanlige trykknappene. Sammenligningen sier samtidig noe om de vanlige trykknappene den måler mot, siden de kommer dårligere ut av den. Det gir fire vurderinger.

Step 2: Setningen navngir ingen som står bak vurderingene; "vi" i den siste leddsetningen viser til hvem som finner de vanlige trykknappene på andre skjermer, ikke til noen som framsetter vurderingen. Source = [] i alle fire.

Step 3: Den første gjelder "knappene", de to neste "hele oppsettet" — helheten som gir kontrollen og som sammenlignes — og den siste "de vanlige tryknappene", som er det sammenligningen faller i disfavør av. Bestemmelsen "vanlige" må være med, ellers er det uklart hvilke trykknapper det siktes til.

Step 4: "plassert pÃ¥ en utmerket mÃ¥te" er hele uttrykket for den første, siden det er måten plasseringen er løst på som roses. "gir god kontroll" er den andre. For sammenligningen trengs hele strekket fra "langt mer behagelig i bruk" til og med leddsetningen om hvor de vanlige trykknappene finnes, fordi det er nettopp bredden i sammenligningsgrunnlaget — de aller fleste skjermer — som gir rosen tyngde. Sett fra de vanlige trykknappenes side er det derimot bare den komparative delen til og med "enn" som rammer dem, altså det som plasserer dem under.

Step 5: God plassering, god kontroll og klart mer behagelig bruk er alle fordeler ved denne skjermen: Positive. Den samme sammenligningen setter de vanlige trykknappene i et dårlig lys, siden de nettopp er det mindre behagelige alternativet: Negative.

Step 6: Fire tupler for de fire vurderingene som ble skilt ut, der den samme sammenligningen er ført opp én gang for hver av de to størrelsene den rammer, og ikke som en dublett. Innledningen er holdt utenfor. Alle fragmentene finnes ordrett i setningen, også skrivemåten "tryknappene" slik den står.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["knappene"],
                    "Polar_expression": ["plassert pÃ¥ en utmerket mÃ¥te"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["hele oppsettet"],
                    "Polar_expression": ["gir god kontroll"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["hele oppsettet"],
                    "Polar_expression": [
                        "langt mer behagelig i bruk enn de vanlige tryknappene som vi finner pÃ¥ de aller fleste skjermer"
                    ],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["de vanlige tryknappene"],
                    "Polar_expression": ["langt mer behagelig i bruk enn"],
                    "Polarity": "Negative",
                },
            ]
        },
    },
    {
        "text": """En mer allsidig og tilkoblingsvennlig skjerm har vi knapt sett .""",
        "reasoning": """Step 1: Setningen er én ytring: en påstand om at man nesten aldri har sett noe bedre av dette slaget. Det er én vurdering.

Step 2: Den som uttaler seg, står eksplisitt i setningen som "vi", altså redaksjonen bak omtalen. Source = ["vi"].

Step 3: Skjermen som roses, nevnes ikke som en egen størrelse her — ordet "skjerm" inngår i den generelle beskrivelsen av hva man knapt har sett, altså i selve sammenligningsklassen, og ikke som en peker til produktet som vurderes. Target = [].

Step 4: Hele setningen fram til punktum utgjør vurderingen: det er kombinasjonen av de to egenskapene og påstanden om at man knapt har sett noe slikt som skaper rosen. Kuttes den opp, forsvinner enten egenskapene eller den negasjonsbaserte sammenligningen som gjør dette til toppkarakter.

Step 5: Å si at man knapt har sett noe mer allsidig og tilkoblingsvennlig er å plassere produktet helt i toppen; negasjonen peker her oppover, ikke nedover. Polarity = Positive.

Step 6: Én tuppel for den ene ytringen. Fragmentet er kontrollert mot setningen og stemmer ordrett, og Target står tomt fordi produktet ikke navngis her.""",
        "output": {
            "opinions": [
                {
                    "Source": ["vi"],
                    "Target": [],
                    "Polar_expression": [
                        "En mer allsidig og tilkoblingsvennlig skjerm har vi knapt sett"
                    ],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """Skjermen kan bare vippes , ikke heves / senkes eller dreies .""",
        "reasoning": """Step 1: Setningen sier hva skjermen kan justeres for, og hva den ikke kan. Det er to sider av samme begrensning, men de bærer ulikt innhold: den ene sier at bevegelsen er avgrenset til én type, den andre navngir de justeringene som mangler. To vurderinger, som deler modalverbet.

Step 2: Ingen står oppført som den som vurderer; begrensningen slås fast som en egenskap. Source = [] i begge.

Step 3: Begge gjelder "Skjermen", som er det eneste som omtales. Target = ["Skjermen"] i begge.

Step 4: Den første er "kan bare vippes", der "bare" er det avgjørende ordet — uten det ville setningen framheve en mulighet i stedet for en avgrensning. Den andre består av "kan" og "ikke heves / senkes eller dreies": modalverbet står bare én gang i setningen men hører grammatisk til også den nektede delen, og oppregningen må være hel, siden hver av de tre justeringene som mangler er en egen mangel. Derfor overlapper de to uttrykkene i modalverbet.

Step 5: Begrenset justeringsmulighet er en ulempe for en skjerm som skal tilpasses arbeidsplassen, og oppregningen av det som ikke går, forsterker det samme. Polarity = Negative i begge.

Step 6: To tupler for de to sidene av begrensningen, ikke én slått sammen og ikke tre. Alle fragmentene finnes ordrett i setningen, skråstreken mellom de to verbene inkludert.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["Skjermen"],
                    "Polar_expression": ["kan bare vippes"],
                    "Polarity": "Negative",
                },
                {
                    "Source": [],
                    "Target": ["Skjermen"],
                    "Polar_expression": ["kan", "ikke heves / senkes eller dreies"],
                    "Polarity": "Negative",
                },
            ]
        },
    },
    {
        "text": """Vi ble positivt overrasket over bildekvaliteten .""",
        "reasoning": """Step 1: Setningen rapporterer én reaksjon: overraskelse over en bestemt egenskap ved produktet. Én vurdering.

Step 2: De som reagerer, står først i setningen som "Vi". Det er et eksplisitt ord i teksten og ikke noe som må sluttes ut. Source = ["Vi"].

Step 3: Det reaksjonen gjelder, er "bildekvaliteten", nevnt én gang og uten kortere alternativ. Target = ["bildekvaliteten"].

Step 4: Uttrykket er "positivt overrasket": adverbet er nødvendig, fordi overraskelse alene ikke sier i hvilken retning forventningene ble brutt. Preposisjonsleddet peker på det vurderte og hører til der.

Step 5: Å bli positivt overrasket over bildekvaliteten betyr at den overgikk forventningene. Polarity = Positive.

Step 6: Én tuppel for den ene reaksjonen, med alle fire feltene. Alle fragmentene er ordrette utsnitt av setningen.""",
        "output": {
            "opinions": [
                {
                    "Source": ["Vi"],
                    "Target": ["bildekvaliteten"],
                    "Polar_expression": ["positivt overrasket"],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """Spesielt fargedynamikken var klart over gjennomsnittet .""",
        "reasoning": """Step 1: Setningen løfter fram én egenskap og plasserer den i forhold til et normalnivå. Én vurdering.

Step 2: Ingen som uttaler seg er nevnt; plasseringen slås fast direkte. Source = [].

Step 3: Egenskapen som vurderes, er "fargedynamikken". Det innledende adverbet framhever den blant andre egenskaper, men er ikke en del av navnet på det som måles. Target = ["fargedynamikken"].

Step 4: Vurderingen er "klart over gjennomsnittet": både retningen og avstanden trengs, siden det nettopp er å ligge tydelig over normalen som utgjør rosen. Kopula holdes utenfor som ren grammatikk.

Step 5: Å ligge klart over gjennomsnittet er en fordel, og framhevingen innledningsvis viser at dette regnes som et av produktets sterke punkter. Polarity = Positive.

Step 6: Én tuppel for den ene vurderingen. Fragmentene stemmer ordrett med setningen og kan ikke kortes ned uten å miste sammenligningen.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["fargedynamikken"],
                    "Polar_expression": ["klart over gjennomsnittet"],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """Det er flere faste fargeprofiler med forskjellige temperaturer Ã¥ velge mellom , i tillegg til manuell innstilling .""",
        "reasoning": """Step 1: Setningen framhever tre ting ved det som tilbys: at profilene er mange, at de dekker ulike temperaturer man kan velge mellom, og at man i tillegg kan stille inn manuelt. Antallet, spennvidden og den manuelle muligheten er tre forskjellige fortrinn ved samme tilbud, ikke omskrivinger av hverandre.

Step 2: Ingen som vurderer er nevnt i setningen. Source = [] i alle tre.

Step 3: Alle tre gjelder utvalget som beskrives, nevnt som "faste fargeprofiler"; den manuelle innstillingen presenteres nettopp som noe som kommer i tillegg til disse profilene og utvider samme tilbud. Target = ["faste fargeprofiler"] i alle tre.

Step 4: "flere" er det ordet som sier at utvalget er romslig. "forskjellige temperaturer Ã¥ velge mellom" må være helt med: det er kombinasjonen av variasjon og valgmulighet som gjør spennvidden til et fortrinn. "manuell innstilling" er den tredje muligheten, navngitt for seg.

Step 5: For den som vil tilpasse bildet, er både antall profiler, ulike fargetemperaturer og muligheten til å stille inn selv fordeler ved skjermen. Polarity = Positive i alle tre.

Step 6: Tre tupler for de tre fortrinnene som ble telt opp, ingen utelatt og ingen dublett. Alle fragmentene er kontrollert mot setningen og finnes der ordrett, med tegnene slik de står i teksten.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["faste fargeprofiler"],
                    "Polar_expression": ["flere"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["faste fargeprofiler"],
                    "Polar_expression": ["forskjellige temperaturer Ã¥ velge mellom"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["faste fargeprofiler"],
                    "Polar_expression": ["manuell innstilling"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
],
    "multibooked_ca": [
    {
        "text": """Bona ubicaciÃ³ .""",
        "reasoning": """Step 1: És un fragment nominal de valoració, sense verb: un adjectiu qualifica un aspecte de l'allotjament. Hi ha una sola opinió.

Step 2: No apareix cap pronom ni cap nom que indiqui qui fa la valoració; el fragment només enuncia la qualitat. Source = [].

Step 3: L'aspecte valorat és el lloc on es troba l'establiment, anomenat "ubicaciÃ³". És l'única menció de l'element avaluat i no admet cap escurçament. Target = ["ubicaciÃ³"].

Step 4: Tota la càrrega valorativa recau en l'adjectiu "Bona", que és el que atribueix la qualitat. No hi ha intensificadors ni cap altre element que calgui incloure.

Step 5: Qualificar de bona la situació d'un hotel és un elogi directe, sense ironia ni negació. Polarity = Positive.

Step 6: Una opinió, una tupla, amb els quatre camps. Els dos fragments extrets coincideixen exactament amb el text original, accents i grafia inclosos, i cap dels dos es pot reduir més.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["ubicaciÃ³"],
                    "Polar_expression": ["Bona"],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """Tot molt net i nou .""",
        "reasoning": """Step 1: El fragment atribueix dues qualitats coordinades a un mateix element: la netedat i el fet de ser nou. Són propietats independents — una cosa pot ser neta sense ser nova i a l'inrevés — de manera que hi ha dues opinions sobre el mateix element.

Step 2: No hi ha cap paraula que identifiqui qui ho diu. Source = [] en totes dues.

Step 3: L'element valorat és el quantificador global "Tot", que és la manera com el fragment es refereix al conjunt de l'allotjament. Serveix de Target per a les dues tuples.

Step 4: La primera valoració és "molt net": l'intensificador forma part de la força amb què s'afirma la netedat. La segona és "nou", que aporta una informació qualitativa diferent; la conjunció que les uneix no aporta contingut avaluatiu.

Step 5: En una ressenya d'allotjament, tant la netedat com el fet que les instal·lacions siguin noves es presenten com a avantatges. Polarity = Positive en tots dos casos.

Step 6: Dues tuples per a les dues qualitats detectades al principi, sense confondre-les en una de sola ni afegir-ne cap. Tots els fragments són subcadenes exactes del text.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["Tot"],
                    "Polar_expression": ["molt net"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["Tot"],
                    "Polar_expression": ["nou"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
    {
        "text": """Agafeu plaÃ§a de parquing a el Catalunya perque gairebe es impossible aparcar a el carrer .""",
        "reasoning": """Step 1: La frase dona un consell i tot seguit n'exposa el motiu. El consell d'agafar plaça de pàrquing no valora res per si mateix: la valoració és la dificultat que el justifica, introduïda per "perque". Hi ha una sola opinió.

Step 2: L'imperatiu s'adreça al lector i no hi ha cap paraula que designi qui parla. Source = [].

Step 3: El que es valora com a problemàtic és l'acció d'aparcar fora, expressada com "aparcar a el carrer". Cal mantenir el complement de lloc, perquè la dificultat s'afirma justament del carrer i no de l'aparcament en general. Target = ["aparcar a el carrer"].

Step 4: L'expressió que porta la valoració és "gairebe es impossible". L'adverbi de matís no es pot deixar fora, perquè és el que converteix una impossibilitat absoluta en una dificultat gairebé total, que és el que s'afirma realment.

Step 5: Que sigui pràcticament impossible aparcar al carrer es presenta com un inconvenient de la zona, i és per això que es recomana reservar plaça. Polarity = Negative.

Step 6: Una tupla per a l'única valoració; el consell inicial s'ha deixat fora en lloc de convertir-lo en una segona opinió. Els fragments extrets es corresponen caràcter per caràcter amb la frase, amb la grafia sense accents tal com hi apareix.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["aparcar a el carrer"],
                    "Polar_expression": ["gairebe es impossible"],
                    "Polarity": "Negative",
                }
            ]
        },
    },
    {
        "text": """La dutxa estÃ  en alt i no hi ha bidet No hi ha mirall per veures sencer No hi ha ascensor El llit , molt grand i comoda El bufet petit perÃ² complert""",
        "reasoning": """Step 1: El text és una llista de punts encadenats sense puntuació que separi clarament les frases. S'hi distingeixen: la posició elevada de la dutxa, l'absència de bidet, l'absència de mirall, l'absència d'ascensor, dues qualitats del llit i dues del bufet. Cada mancança afecta un element diferent i cada qualitat del llit i del bufet aporta una informació distinta, de manera que cap d'elles repeteix una altra.

Step 2: No hi ha cap pronom ni nom que indiqui qui fa les observacions; totes s'enuncien com a fets de l'allotjament. Source = [] en totes les tuples.

Step 3: Els elements valorats són, cadascun amb la forma exacta amb què apareixen: "ascensor", "El bufet" (dues vegades, per a les dues qualitats), "El llit" (també dues vegades), "La dutxa", "bidet" i "mirall". Els articles s'inclouen només allà on formen part de la menció tal com el text la fa.

Step 4: Les mancances es marquen amb la construcció existencial negada, que apareix en majúscula al començament de punt i en minúscula després de la conjunció; per això el bidet porta "no hi ha" i el mirall i l'ascensor porten "No hi ha". Del bufet es diu "petit", que és el retret, i "complert", que és l'elogi. Del llit, "molt grand" i "comoda". De la dutxa, el retret és que està situada en un pla elevat, i el fragment retingut és la paraula que introdueix aquesta posició, "en".

Step 5: La manca d'ascensor, de bidet i de mirall són mancances d'elements que un hoste espera trobar, i un bufet petit és una limitació: Negative. Que el bufet sigui complet, i que el llit sigui gran i còmode, són avantatges: Positive. La ubicació elevada de la dutxa es presenta com una incomoditat: Negative.

Step 6: Vuit tuples per als vuit punts detectats a l'inici, sense que se n'hagi omès cap ni s'hagi duplicat cap. Tots els fragments s'han comprovat contra el text i hi apareixen exactament, incloses les majúscules i les grafies no normatives com "grand" i "comoda".""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["ascensor"],
                    "Polar_expression": ["No hi ha"],
                    "Polarity": "Negative",
                },
                {
                    "Source": [],
                    "Target": ["El bufet"],
                    "Polar_expression": ["petit"],
                    "Polarity": "Negative",
                },
                {
                    "Source": [],
                    "Target": ["El bufet"],
                    "Polar_expression": ["complert"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["El llit"],
                    "Polar_expression": ["comoda"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["El llit"],
                    "Polar_expression": ["molt grand"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["La dutxa"],
                    "Polar_expression": ["en"],
                    "Polarity": "Negative",
                },
                {
                    "Source": [],
                    "Target": ["bidet"],
                    "Polar_expression": ["no hi ha"],
                    "Polarity": "Negative",
                },
                {
                    "Source": [],
                    "Target": ["mirall"],
                    "Polar_expression": ["No hi ha"],
                    "Polarity": "Negative",
                },
            ]
        },
    },
    {
        "text": """- Habitacions petites perÃ² suficients per un cap de setmana . - A el costat de les 2 Ãºniques discoteques que hi ha a cadaquÃ¨s , per_tant tot_i_que les habitacions estan forÃ§a insonoritzades segueix sentint-se soroll .""",
        "reasoning": """Step 1: El primer punt conté un retret i una concessió sobre les habitacions: són petites, però n'hi ha prou per a una estada curta. El segon punt situa l'establiment al costat de les discoteques i, després de reconèixer que les habitacions estan ben aïllades, acaba constatant que el soroll s'acaba sentint igual. Es distingeixen doncs quatre valoracions: la mida, la suficiència, l'aïllament i el soroll que persisteix.

Step 2: Cap fragment no designa qui fa les observacions. Source = [] en totes les tuples.

Step 3: La mida i la suficiència es prediquen de les "Habitacions" del primer punt, tal com hi apareixen escrites; l'aïllament es predica de "les habitacions" del segon punt, amb l'article que hi duu. En canvi, la constatació final del soroll no predica res d'una entitat anomenada: el soroll és allò que se sent, no l'element avaluat, i les discoteques són l'origen del problema, no el que es valora. Target = [] en aquella tupla.

Step 4: "segueix sentint-se soroll" cal mantenir-ho sencer, perquè el verb "segueix" és el que expressa que el problema persisteix malgrat l'aïllament. "suficients per un cap de setmana" també es manté sencer: la suficiència s'afirma només per a una estada breu, i sense aquesta acotació la valoració canviaria de sentit. "petites" és el retret nu, i "forÃ§a insonoritzades" inclou el quantificador perquè és el que gradua l'aïllament reconegut.

Step 5: Habitacions petites són un inconvenient, i el soroll que continua sentint-se també: Negative. Que resultin suficients per a un cap de setmana i que estiguin prou insonoritzades es diuen a favor de l'allotjament: Positive.

Step 6: Quatre tuples per a les quatre valoracions comptades al principi. Els connectors escrits amb guionets baixos no s'han inclòs en cap fragment perquè només articulen el raonament. Tots els fragments s'han verificat contra el text i hi apareixen exactament.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": [],
                    "Polar_expression": ["segueix sentint-se soroll"],
                    "Polarity": "Negative",
                },
                {
                    "Source": [],
                    "Target": ["Habitacions"],
                    "Polar_expression": ["suficients per un cap de setmana"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["Habitacions"],
                    "Polar_expression": ["petites"],
                    "Polarity": "Negative",
                },
                {
                    "Source": [],
                    "Target": ["les habitacions"],
                    "Polar_expression": ["forÃ§a insonoritzades"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
    {
        "text": """Per sort tenen taps de les orelles . - Ubicat molt cÃ¨ntric - Personal molt amable .""",
        "reasoning": """Step 1: La primera frase agraeix que l'establiment posi taps per a les orelles a disposició dels hostes. El punt sobre la ubicació es limita a situar l'establiment al centre, i el darrer punt qualifica el personal. Les valoracions pròpiament dites són la disponibilitat dels taps i el tracte del personal.

Step 2: Cap de les dues frases no designa qui parla; el "Per sort" expressa alleujament però no anomena ningú. Source = [] en totes dues.

Step 3: L'amabilitat es predica del "Personal", que és l'element valorat. En canvi, el fet de tenir taps no predica cap qualitat d'una entitat anomenada: el subjecte queda implícit en la desinència verbal i els taps són el mitjà que resol el problema, no allò que s'avalua. Target = [] en aquella tupla.

Step 4: "tenen taps de les orelles" es manté sencer perquè la valoració consisteix justament en el fet de disposar-ne: ni el verb sol ni el nom sol expressarien que l'establiment els proporciona. La locució inicial d'alleujament no s'inclou perquè només marca l'actitud amb què s'acull el fet. Del personal, la valoració és "molt amable", amb l'intensificador que en gradua l'elogi.

Step 5: Que hi hagi taps disponibles es presenta com una sort per a l'hoste, i un personal molt amable és un elogi clar. Polarity = Positive en tots dos casos.

Step 6: Dues tuples per a les dues valoracions identificades; el punt sobre on està situat l'establiment no s'ha convertit en una opinió addicional. Tots els fragments coincideixen exactament amb el text.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": [],
                    "Polar_expression": ["tenen taps de les orelles"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["Personal"],
                    "Polar_expression": ["molt amable"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
    {
        "text": """El bany es una mica senzill La vista desde la finestra""",
        "reasoning": """Step 1: El text conté una predicació sobre el bany i, a continuació, un fragment nominal sense verb ni adjectiu, que només anomena la vista des de la finestra sense dir-ne res. Només la primera part expressa una valoració, de manera que hi ha una sola opinió.

Step 2: No hi ha cap paraula que identifiqui qui fa l'observació. Source = [].

Step 3: L'element valorat és "El bany", amb l'article amb què el text l'anomena. Target = ["El bany"].

Step 4: La valoració és "una mica senzill": el matisador atenua el retret i en forma part, perquè el que s'afirma no és que el bany sigui senzill sense més, sinó que ho és fins a cert punt. La còpula queda fora perquè només fa d'enllaç gramatical.

Step 5: Qualificar un bany de senzill en una ressenya és assenyalar-hi una manca de prestacions, encara que sigui amb reserves. Polarity = Negative.

Step 6: Una tupla per a l'única predicació valorativa; el fragment final s'ha deixat fora perquè no hi predica res. Els dos fragments extrets són subcadenes exactes del text, amb la grafia tal com hi apareix.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["El bany"],
                    "Polar_expression": ["una mica senzill"],
                    "Polarity": "Negative",
                }
            ]
        },
    },
    {
        "text": """M' ha agradat tot La situaciÃ³ i la relaciÃ³ qualitat preu""",
        "reasoning": """Step 1: El text expressa que a qui escriu li ha agradat una cosa, i tot seguit enumera els elements que li han agradat: el conjunt, la situació i la relació qualitat-preu. Cadascun d'aquests elements és un objecte d'apreciació diferent, de manera que hi ha tres opinions que comparteixen la mateixa expressió d'agrado.

Step 2: L'única marca de qui fa la valoració és el pronom feble "M'", que apareix explícitament al començament. És curt, però és la traça textual del qui experimenta l'agrado, i serveix per a les tres tuples. Source = ["M'"] en totes.

Step 3: Els tres elements apreciats són "la relaciÃ³ qualitat preu", "tot" i "La situaciÃ³", cadascun amb la forma i les majúscules amb què apareix al text. Cap d'ells no es pot escurçar sense deixar d'identificar l'element.

Step 4: L'expressió que porta l'apreciació és "ha agradat", el verb que atribueix el gust. Com que una sola predicació cobreix els tres elements enumerats, el mateix fragment fa d'expressió avaluativa en les tres tuples; el pronom feble pertany al camp del qui valora i no s'hi inclou.

Step 5: Dir que una cosa ha agradat és expressar-hi satisfacció, sense cap element que en giri el sentit. Polarity = Positive en les tres.

Step 6: Tres tuples, una per cada element enumerat, i cap no s'ha deixat de banda. Tots els fragments, inclòs el pronom amb l'apòstrof, s'han comprovat contra el text i hi apareixen exactament.""",
        "output": {
            "opinions": [
                {
                    "Source": ["M'"],
                    "Target": ["la relaciÃ³ qualitat preu"],
                    "Polar_expression": ["ha agradat"],
                    "Polarity": "Positive",
                },
                {
                    "Source": ["M'"],
                    "Target": ["tot"],
                    "Polar_expression": ["ha agradat"],
                    "Polarity": "Positive",
                },
                {
                    "Source": ["M'"],
                    "Target": ["La situaciÃ³"],
                    "Polar_expression": ["ha agradat"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
    {
        "text": """La impossibilitat de fer migdiada a les 4 de la tarda , les dones de la neteja criden d' un pis a un altre i se sent tot .""",
        "reasoning": """Step 1: S'hi distingeixen tres queixes: no poder fer la migdiada, el fet que el personal de neteja cridi d'un pis a l'altre, i que se senti tot. La primera enuncia la conseqüència que pateix l'hoste, la segona assenyala una conducta concreta i la tercera es refereix a l'aïllament de l'edifici; són tres queixes diferents i no una repetida.

Step 2: Cap de les tres no s'atribueix a ningú de manera explícita; totes s'enuncien directament. Source = [] en totes.

Step 3: La conducta de cridar es predica de "les dones de la neteja", que és qui la fa i el que es retreu. En canvi, la impossibilitat de dormir i el fet que se senti tot no prediquen res d'una entitat anomenada: cap element de la frase no apareix com allò avaluat en aquestes dues queixes. Target = [] en aquestes dues tuples.

Step 4: "La impossibilitat de fer migdiada" es manté sencer perquè el nom d'impossibilitat sol no diria què és el que no es pot fer, i és justament la migdiada frustrada el que constitueix la queixa; l'hora concreta queda fora perquè només la situa en el temps. "criden" és el verb que expressa la conducta retreta, sense el complement de lloc, que només diu d'on a on. "se sent tot" es manté sencer perquè el quantificador és el que expressa la manca total d'aïllament.

Step 5: No poder descansar, que el personal cridi i que se senti tot des de l'habitació són tres inconvenients de l'estada. Polarity = Negative en les tres.

Step 6: Tres tuples per a les tres queixes comptades a l'inici, sense convertir cap complement circumstancial en una opinió de més. Tots els fragments són subcadenes exactes de la frase i cap no es pot reduir sense perdre el sentit de la queixa.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": [],
                    "Polar_expression": ["La impossibilitat de fer migdiada"],
                    "Polarity": "Negative",
                },
                {
                    "Source": [],
                    "Target": ["les dones de la neteja"],
                    "Polar_expression": ["criden"],
                    "Polarity": "Negative",
                },
                {
                    "Source": [],
                    "Target": [],
                    "Polar_expression": ["se sent tot"],
                    "Polarity": "Negative",
                },
            ]
        },
    },
    {
        "text": """Un de els dos dies que ens vam allotjar van trigar mÃ©s de mitja hora a donar-nos l' habitaciÃ³ , encara que havÃ­em trucat a el matÃ­ per a demanar-ho L' esmorzar""",
        "reasoning": """Step 1: El text conté una queixa: un dels dos dies van tardar molt a donar l'habitació, tot i haver-la demanada al matí. La subordinada concessiva reforça la queixa explicant per què la tardança era menys excusable, i el fragment final només anomena l'esmorzar sense dir-ne res. Hi ha una sola opinió.

Step 2: Les formes verbals de primera persona del plural ("ens vam allotjar", "havÃ­em trucat") pertanyen als fets que emmarquen la queixa — qui s'allotjava i qui havia trucat — i no hi ha cap paraula que presenti algú com qui emet la valoració de la tardança. Source = [].

Step 3: La queixa no predica cap qualitat d'una entitat anomenada al text: el subjecte que va tardar queda implícit en la desinència verbal, i l'habitació és allò que es va lliurar tard, no l'element avaluat. Target = [].

Step 4: L'expressió de la queixa és "van trigar mÃ©s de mitja hora a donar-nos l' habitaciÃ³". Cal mantenir-la sencera: el verb sol no diria quant es va tardar, la durada sola no diria en què, i és la combinació de les dues coses la que constitueix el retret. La concessiva queda fora perquè justifica la queixa en lloc d'expressar-la.

Step 5: Esperar més de mitja hora per rebre l'habitació es presenta com un mal servei, i la menció de la trucada prèvia hi accentua el retret. Polarity = Negative.

Step 6: Una tupla per a l'única queixa; ni la concessiva ni el fragment final s'han convertit en opinions addicionals. El fragment extret s'ha comprovat contra el text i hi apareix exactament, amb l'apòstrof i l'espai tal com hi consten.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": [],
                    "Polar_expression": [
                        "van trigar mÃ©s de mitja hora a donar-nos l' habitaciÃ³"
                    ],
                    "Polarity": "Negative",
                }
            ]
        },
    },
],
    "multibooked_eu": [
    {
        "text": """Marsella eta ingurua ezagutzeko leku seguru eta lasaiaren bila genbiltzan eta leku hau topatu genuen , gure asmoekin bat egiten zuena .""",
        "reasoning": """Step 1: Esaldiaren lehen zatiak zer bilatzen zen azaltzen du, eta bigarrenak lekua aurkitu zela kontatzen. Balorazioa bilatzen zen lekuari eskatzen zitzaizkion bi ezaugarrietan dago: segurtasuna eta lasaitasuna. Bi ezaugarri horiek elkarrengandik bereizten dira — leku bat lasaia izan daiteke segurua izan gabe —, beraz bi iritzi daude; topatzearen berri ematea, berriz, gertaera bat da eta ez balorazio bat.

Step 2: Ez da izenordainik ageri iritzia daukanaren ordez; pertsona-marka "genbiltzan" aditz-forman bertan dago sartuta, eta hori da esaldiak bilatzailea nor zen adierazteko duen arrasto esplizitu bakarra. Source = ["genbiltzan"] bi iritzietan.

Step 3: Baloratzen den entitatea "leku" da, hots, bilatzen zen tokia, esaldian horrela izendatua. Ez dago izendapen laburragorik, eta bi iritziek entitate bera dute.

Step 4: Ezaugarriak "seguru" eta "lasaiaren" dira, bakoitza bere aldetik; artean doan juntagailuak ez du eduki baloratzailerik gehitzen, eta "bila" hitzak bilaketa markatzen du, ez ezaugarria bera.

Step 5: Ostatu bat bilatzen duenarentzat segurtasuna eta lasaitasuna nahi diren ezaugarriak dira, eta esaldiak berak horiek bilatu zirela dio. Polarity = Positive bietan.

Step 6: Bi tupla, hasieran zenbatu ziren bi ezaugarrietarako, eta topaketaren berria ez da hirugarren iritzi bihurtu. Zati guztiak esaldiaren azpikate zehatzak dira, deklinabide-atzizkiak barne, jatorrizko forman utzita.""",
        "output": {
            "opinions": [
                {
                    "Source": ["genbiltzan"],
                    "Target": ["leku"],
                    "Polar_expression": ["lasaiaren"],
                    "Polarity": "Positive",
                },
                {
                    "Source": ["genbiltzan"],
                    "Target": ["leku"],
                    "Polar_expression": ["seguru"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
    {
        "text": """Nahiz eta hirian bertan egon , erdialdetik nahiko urrun dago .""",
        "reasoning": """Step 1: Esaldiak bi datu jartzen ditu aurrez aurre: hirian bertan egotea eta erdialdetik nahiko urrun geratzea. Kontzesiozko egiturak batak bestearen kontra jokatzen duela erakusten du, eta horrek bi balorazio bereizi ematen ditu.

Step 2: Ez da inor izendatzen esatearen jabe gisa; biak ostatuaren egoerari buruzko datu gisa ematen dira. Source = [] bietan.

Step 3: Esaldian ez da izendatzen zer dagoen hirian edo zer dagoen erdialdetik urrun: subjektua aditz-komunztaduran baino ez da ageri, eta "hirian" eta "erdialdetik" kokapenaren erreferentziak dira, ez baloratzen den entitatea. Target = [] bietan.

Step 4: Lehena "hirian bertan egon" da: "bertan" ezinbestekoa da, horrek baieztatzen baitu hiri barruan bertan dagoela eta ez inguruan. Bigarrena "erdialdetik nahiko urrun dago" osorik behar da: nondik urrun dagoen esan gabe, distantziak ez luke kexaren zentzua izango, eta "nahiko" hitzak zenbateraino graduatzen du.

Step 5: Hiri barruan bertan egotea abantaila gisa aurkezten da, eta horregatik Positive. Erdialdetik urrun geratzea, aldiz, eragozpen gisa, eta kontzesiozko egiturak hori azpimarratzen du: Negative.

Step 6: Bi tupla, hasieran bereizi ziren bi balorazioetarako, bat ere galdu gabe. Zati biak esaldiaren azpikate zehatzak dira eta ezin dira laburtu esanahia galdu gabe. Iturria eta baloratutako entitatea hutsik daude, esaldiak ez baititu izendatzen.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": [],
                    "Polar_expression": ["hirian bertan egon"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": [],
                    "Polar_expression": ["erdialdetik nahiko urrun dago"],
                    "Polarity": "Negative",
                },
            ]
        },
    },
    {
        "text": """Horrek alde txarrak ditu : erdialdera edo hegoaldeko tokietara joateko Marsella kaotikoa gurutzatu behar dela , baina gu lasaitasun bila genbiltzan , baita autoa uzteko leku seguru baten bila ere .""",
        "reasoning": """Step 1: Esaldiak alde txarrak dituela dio, eta bi puntuen ondoren zertan datzan azaltzen du: hiria gurutzatu behar izatea. Azalpen hori eragozpenaren edukia da, ez beste iritzi bat, eta "baina" ondoko zatiak bidaiariaren lehentasunak gogoratzen ditu eragozpena erlatibizatzeko. Iritzi bakarra.

Step 2: Ez dago iritzia norena den adierazten duen hitzik eragozpenaren enuntziatuan; zuzenean baieztatzen da. Source = [].

Step 3: Esaldiak ez du izendatzen zer duen alde txarrak: "Horrek" aurretik esandakora igortzen du, esaldi honen barruan zer den zehaztu gabe, eta hortaz ez dago baloratutako entitatea izendatzen duen zatirik. Target = [].

Step 4: Eragozpena "alde txarrak ditu" adierazpenak darama: izena eta aditza behar dira biak, izenak zer dagoen esaten du eta aditzak nori dagokion lotzen. Ondoko azalpen luzea kanpoan geratzen da, arrazoia ematen baitu eta ez balorazioa bera.

Step 5: Alde txarrak izatea esatea eragozpenak aitortzea da, eta ondoko azalpenak trafikoaren zailtasuna aipatzen du horren froga gisa. Polarity = Negative.

Step 6: Tupla bakarra iritzi bakarrerako; ez azalpena ez lehentasunen aipamena ez dira iritzi gehigarri bihurtu. Ateratako zatia esaldian dagoen bezalaxe agertzen da.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": [],
                    "Polar_expression": ["alde txarrak ditu"],
                    "Polarity": "Negative",
                }
            ]
        },
    },
    {
        "text": """Bestela , etxea sinplea da , baina egokia da .""",
        "reasoning": """Step 1": Esaldiak bi ezaugarri kontrajarri ematen dizkio etxeari: sinplea izatea eta egokia izatea. "baina" juntagailuak berak erakusten du bata bestearen kontrapisua dela, beraz bi iritzi dira eta ez bakarra.

Step 2: Ez da inor aipatzen baieztapenen jabe gisa. Source = [] bietan.

Step 3: Baloratzen den entitatea "etxea" da, esaldian behin bakarrik izendatua eta laburtu ezin dena. Bi tupletan berdina.

Step 4: Ezaugarriak "sinplea" eta "egokia" dira, bakoitza bere adjektiboan; aditz laguntzailea lotura hutsa da eta juntagailuak ez du eduki baloratzailea gehitzen.

Step 5: Ostatuari buruzko iruzkinetan sinplea izatea gabezia moduan aurkezten da — horregatik dator gero kontrapisua —, beraz Negative. Egokia izatea, berriz, aldeko balorazioa da: Positive.

Step 6: Bi tupla, hasieran bereizi ziren bi ezaugarrietarako, bat bestearen barruan sartu gabe. Zati guztiak esaldiaren azpikate zehatzak dira.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["etxea"],
                    "Polar_expression": ["egokia"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["etxea"],
                    "Polar_expression": ["sinplea"],
                    "Polarity": "Negative",
                },
            ]
        },
    },
    {
        "text": """Etxearen orubea nahiko trakets topatu genuen ( beste terrazak , alegia ) , baina justu alokatzen dena oso txukun .""",
        "reasoning": """Step 1: Esaldiak bi gauza bereizten ditu: etxearen orubea, traketsa iruditu zitzaiena, eta alokatzen den zatia, txukun aurkitu zutena. Bi entitate desberdinen gainean bi balorazio kontrajarri dira, "baina" juntagailuak bereizita.

Step 2: Lehen balorazioan pertsona-marka "genuen" aditz-forman dago sartuta, eta hori da esaldiak balorazioa nork egin zuen adierazteko duen arrasto esplizitu bakarra: Source = ["genuen"]. Bigarren zatian aditza elidituta dago eta ez da inolako pertsona-markarik gelditzen, beraz han Source = [].

Step 3: Lehenengoak "Etxearen orubea" du baloratutako entitatea, eta genitiboa beharrezkoa da, orubea zeinena den horrek zehazten baitu. Bigarrenak "alokatzen dena", hots, benetan errentan hartzen den zatia, gainerako lursailetik bereizita.

Step 4: Balorazioak "nahiko trakets" eta "oso txukun" dira, biak graduatzailea barne, horiek adierazten baitute zenbaterainokoa den akatsa batean eta txukuntasuna bestean. Parentesi arteko zehaztapenak zein terrazez ari den argitzen du eta ez du balorazio berririk gehitzen.

Step 5: Orubea trakets samarra izatea akats gisa aipatzen da: Negative. Alokatzen den zatia oso txukun egotea, aldiz, aldeko balorazioa: Positive.

Step 6: Bi tupla, hasieran bereizi ziren bi balorazioetarako, entitate bat besteari egotzi gabe. Zati guztiak esaldian dauden bezala jaso dira, eta bigarren tuplako iturria hutsik dago, han ez baita pertsona-markarik gelditzen.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["alokatzen dena"],
                    "Polar_expression": ["oso txukun"],
                    "Polarity": "Positive",
                },
                {
                    "Source": ["genuen"],
                    "Target": ["Etxearen orubea"],
                    "Polar_expression": ["nahiko trakets"],
                    "Polarity": "Negative",
                },
            ]
        },
    },
    {
        "text": """Izan ere , frantsesez bakarrik daki .""",
        "reasoning": """Step 1: Esaldi laburrak muga bat adierazten du: hizkuntza bakarra jakitea. Balorazio bakarra dago.

Step 2: Ez da izenordainik ez izenik ageri baieztapenaren jabe gisa; hasierako lokuzioak aurrekoarekin lotzen du besterik ez. Source = [].

Step 3: Esaldiak ez du izendatzen nor dakien frantsesez soilik: subjektua aditzaren komunztaduran baino ez da ageri, eta hizkuntzaren izena jakintzaren edukia da, ez baloratzen den entitatea. Target = [].

Step 4: Adierazpena "frantsesez bakarrik daki" osorik da: "bakarrik" da muga sortzen duen hitza, eta hori gabe esaldiak gaitasun bat aitortuko luke muga bat aitortu beharrean.

Step 5: Ostatu batean hizkuntza bakarra egitea bisitariarentzako oztopoa da, eta esaldia horrexegatik dator azalpen gisa. Polarity = Negative.

Step 6: Tupla bakarra muga bakarrerako, lau eremuak beteta. Ateratako zatia esaldiaren azpikate zehatza da eta ezin da laburtu mugaren zentzua galdu gabe.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": [],
                    "Polar_expression": ["frantsesez bakarrik daki"],
                    "Polarity": "Negative",
                }
            ]
        },
    },
    {
        "text": """Ingurune eder eta lasaia da .""",
        "reasoning": """Step 1: Esaldiak bi ezaugarri ematen dizkio inguruari: edertasuna eta lasaitasuna. Bereiziak dira, batak itxurari eta besteak zarata-ezari egiten baitiote erreferentzia, beraz bi iritzi.

Step 2: Ez dago balorazioa norena den adierazten duen hitzik. Source = [] bietan.

Step 3: Baloratzen den entitatea "Ingurune" da, esaldiak inguruari erreferentzia egiteko duen hitza. Bi tupletan berdina.

Step 4: Ezaugarriak "eder" eta "lasaia" dira, bakoitza bere aldetik hartuta; juntagailuak eta aditzak ez dute eduki baloratzailea gehitzen.

Step 5: Ostatu baten inguruaz ari denean, ederra eta lasaia izatea bi abantaila dira. Polarity = Positive bietan.

Step 6: Bi tupla bi ezaugarrietarako, bat ere galdu gabe eta bakar batean bildu gabe. Zati guztiak esaldian dauden bezala agertzen dira, atzizkiak barne.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["Ingurune"],
                    "Polar_expression": ["eder"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["Ingurune"],
                    "Polar_expression": ["lasaia"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
    {
        "text": """Etxe zoragarria dute ikuspegi izugarriekin eta mendian ibiltzeko aukera asko dituenak .""",
        "reasoning": """Step 1: Esaldiak hiru gauza azpimarratzen ditu etxeaz: berez zoragarria izatea, ikuspegi izugarriak izatea eta mendian ibiltzeko aukera ugari eskaintzea. Hirurak eduki desberdina dute — bata etxearen beraren balorazioa da, besteak zer eskaintzen duen —, beraz hiru iritzi.

Step 2: Ez da inor izendatzen balorazioen jabe gisa; aditzaren komunztadurak etxearen jabeak aipatzen ditu, baina horiek etxea dutenak dira, ez balorazioa egiten dutenak. Source = [] hiruretan.

Step 3: Hiruren entitatea "Etxe" da, esaldiak baloratzen duen ostatua. Ikuspegiak eta mendiko aukerak etxeak eskaintzen dituen gauzak dira, eta horregatik balorazioaren parte, ez baloratutako entitate desberdinak.

Step 4: Lehena "zoragarria" adjektiboa da. Bigarrena "mendian ibiltzeko aukera asko dituenak" osorik behar da: "asko" da ugaritasuna adierazten duena, eta zertarako diren aukerak esan gabe ez litzateke jakingo zergatik den hori abantaila. Hirugarrena "ikuspegi izugarriekin" da, soziatiboa barne, horrek lotzen baititu ikuspegiak etxearekin.

Step 5: Etxea zoragarria izatea, ikuspegi izugarriak izatea eta mendian ibiltzeko aukera asko izatea hirurak dira ostatuaren alde jokatzen duten gauzak. Polarity = Positive hiruretan.

Step 6: Hiru tupla hasieran zenbatutako hiru balorazioetarako, bat ere kanpoan utzi gabe. Zati guztiak esaldiaren azpikate zehatzak dira eta ezin dira gehiago laburtu.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["Etxe"],
                    "Polar_expression": ["zoragarria"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["Etxe"],
                    "Polar_expression": ["mendian ibiltzeko aukera asko dituenak"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["Etxe"],
                    "Polar_expression": ["ikuspegi izugarriekin"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
    {
        "text": """Logela goxoak , ohe zoragarria eta sukalde amankomun bat dute .""",
        "reasoning": """Step 1: Esaldiak hiru elementu zerrendatzen ditu, bakoitza bere ezaugarriarekin: logelak, ohea eta sukaldea. Elementu desberdinak dira eta bakoitzak bere balorazioa darama, beraz hiru iritzi.

Step 2: Ez dago balorazioak norenak diren adierazten duen hitzik; zerrenda ostatuak duenaren berri gisa ematen da. Source = [] hiruretan.

Step 3: Baloratutako entitateak "Logela", "ohe" eta "sukalde" dira, bakoitza bere zerrenda-atalean izendatuta eta bakoitza bere tuplan.

Step 4: Ezaugarriak "goxoak", "zoragarria" eta "amankomun" dira. Hirugarrenean, sukaldeaz esaten den gauza bakarra erabilera partekatua dela da, eta horixe da hain zuzen bisitariarentzat aipagarria dena: sukalde bat erabilgarri izatea.

Step 5: Logela goxoak eta ohe zoragarria goraipamen zuzenak dira. Sukalde amankomuna izatea ere aldeko datu gisa ematen da, bisitariak janaria prestatzeko aukera baitu horrekin. Polarity = Positive hiruretan.

Step 6: Hiru tupla zerrendako hiru elementuetarako, bat ere galdu gabe eta batzuk besteekin nahastu gabe. Zati guztiak esaldian dauden bezalaxe jaso dira, pluralaren atzizkia barne.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["Logela"],
                    "Polar_expression": ["goxoak"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["ohe"],
                    "Polar_expression": ["zoragarria"],
                    "Polarity": "Positive",
                },
                {
                    "Source": [],
                    "Target": ["sukalde"],
                    "Polar_expression": ["amankomun"],
                    "Polarity": "Positive",
                },
            ]
        },
    },
    {
        "text": """Bisitari guztiontzako gozari bikaina prestatzen dute mahai amankomun batean .""",
        "reasoning": """Step 1: Esaldiak gosaria prestatzen dutela kontatzen du, eta balorazioa gosariari ematen zaion kalifikazioan dago. Non prestatzen den eta norentzat den zirkunstantziak dira. Iritzi bakarra.

Step 2: Aditzaren komunztadurak ostatuko arduradunak aipatzen ditu prestatzaile gisa, baina esaldiak ez du izendatzen balorazioa egiten duen inor. Source = [].

Step 3: Baloratutako entitatea "gozari" da, hots, prestatzen duten gosaria. Ez dago izendapen laburragorik eta ez du beste hautagai hoberik. Target = ["gozari"].

Step 4: Balorazioa "bikaina" adjektiboan dago osorik. Hartzaileen aipamena eta mahaiaren zehaztapena testuinguru hutsak dira balorazioaren aldetik.

Step 5: Gosaria bikaina dela esatea goraipamen zuzena da, inolako ironiarik gabe. Polarity = Positive.

Step 6: Tupla bakarra balorazio bakarrerako, lau eremuak beteta; ez zirkunstantziek ez prestatzeko aipamenak ez dute iritzi gehigarririk sortu. Bi zatiak esaldiaren azpikate zehatzak dira.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["gozari"],
                    "Polar_expression": ["bikaina"],
                    "Polarity": "Positive",
                }
            ]
        },
    },
],
    "mpqa": [
    {
        "text": """The spreading AIDS epidemic is threatening Asia's economic growth , warned officials from the United Nations Program on HIV / AIDS ( UNAIDS ) , the World Health Organization and bank delegates at " Investing in Asia's Health " , an Asian Development Bank ( ADB ) seminar held here Thursday .""",
        "reasoning": """Step 1: The sentence is a single reported warning: a claim about the epidemic and Asian growth, attributed to a group of officials and delegates by the verb "warned". That one speech act, however, is directed at two things of different size — the epidemic itself as an entity, and the whole proposition that it is threatening Asia's economic growth — and each of these is a distinct thing the officials were warning about, so two opinions are extracted from the one verb.

Step 2: The holder is stated after the verb as the coordinated group of officials and delegates. The span must run from "officials" through the appositive naming the seminar, because the delegates are identified only as the ones attending that particular event; cutting the venue would leave it unclear which delegates are meant. Only the closing detail of when it was held falls outside, since that dates the event rather than identifying the speakers. The same holder serves both opinions.

Step 3: For the first opinion the thing warned about is the phenomenon itself, "The spreading AIDS epidemic". For the second it is the full claim, "The spreading AIDS epidemic is threatening Asia's economic growth", which cannot be shortened without losing the specific consequence the officials were warning of. These are two different objects of concern, not two names for one.

Step 4: In both cases the evaluative work is done by "warned": a neutral report would have used "said", and this verb marks the statement as an alert about something harmful. Nothing else in the sentence is needed, so the same one-word span serves both tuples.

Step 5: A warning presents its object as damaging or dangerous, and here what is flagged is harm to regional growth. Polarity = Negative for both.

Step 6: Two tuples, matching the two objects of the single warning identified at the start. Each has a Source, a Target, a Polar_expression and a Polarity, and every span — including the long holder phrase with its internal quotation marks — was checked against the sentence and matches it exactly.""",
        "output": {
            "opinions": [
                {
                    "Source": [
                        "officials from the United Nations Program on HIV / AIDS ( UNAIDS ) , the World Health Organization and bank delegates at \" Investing in Asia's Health \" , an Asian Development Bank ( ADB ) seminar"
                    ],
                    "Target": ["The spreading AIDS epidemic"],
                    "Polar_expression": ["warned"],
                    "Polarity": "Negative",
                },
                {
                    "Source": [
                        "officials from the United Nations Program on HIV / AIDS ( UNAIDS ) , the World Health Organization and bank delegates at \" Investing in Asia's Health \" , an Asian Development Bank ( ADB ) seminar"
                    ],
                    "Target": [
                        "The spreading AIDS epidemic is threatening Asia's economic growth"
                    ],
                    "Polar_expression": ["warned"],
                    "Polarity": "Negative",
                },
            ]
        },
    },
    {
        "text": """Sachs also addressed the exacerbating condition of the AIDS pandemic in China , saying that China had the technical ability to control AIDS infection , but still needed a comprehensive strategy , including local surveillance network , increasing government subsidiary and free medicine for the poor .""",
        "reasoning": """Step 1: The sentence records one act of taking up a topic, marked by "also addressed". What follows "saying that" relays the substance of his remarks — a technical capability and a list of measures still required — as reported content rather than as further attitudinal wording of its own. One opinion.

Step 2: The speaker is named outright at the head of the sentence as "Sachs", which is the whole of what the sentence gives us about who is talking. Source = ["Sachs"].

Step 3: What he took up is the epidemic situation in that country, and the entity itself is named as "the AIDS pandemic in China". The country modifier has to stay, since it is what limits the topic to China rather than the pandemic at large; the leading words about a worsening condition describe the state that topic is in, not the topic. Target = ["the AIDS pandemic in China"].

Step 4: "also addressed" is the span that reports his engagement with the subject: "addressed" states that he spoke to it, and "also" places this alongside his other remarks as part of the same act of speaking. Neither part asserts approval or disapproval of anything.

Step 5: The verb reports that a subject was raised and discussed without revealing any stance toward it, and the reported content that follows is presented as description of circumstances rather than praise or blame. Polarity = Neutral.

Step 6: One tuple for the single act of address, with the reported remarks left as content rather than promoted to extra opinions. All three spans are exact, minimal substrings of the sentence.""",
        "output": {
            "opinions": [
                {
                    "Source": ["Sachs"],
                    "Target": ["the AIDS pandemic in China"],
                    "Polar_expression": ["also addressed"],
                    "Polarity": "Neutral",
                }
            ]
        },
    },
    {
        "text": """In 1998 , MCCOY , Inc. led two efforts that , we believe , will help our community continue to take the steps necessary to help all young people grow up and develop well .""",
        "reasoning": """Step 1: The backbone of the sentence — an organization led two efforts in a given year — is a record of what happened. The evaluative element is the parenthetical assertion of belief about what those efforts will accomplish, which is one opinion.

Step 2: The belief is voiced in the first person plural, and the sentence identifies who that plural refers to by naming the organization presenting these efforts, "MCCOY , Inc." — that is the explicit textual identification of the holder speaking here. Source = ["MCCOY , Inc."].

Step 3: The thing the belief is about is what the organization did, referred to as "two efforts"; the relative clause that follows states what is expected of them rather than naming them. No shorter phrase identifies them. Target = ["two efforts"].

Step 4: The attitudinal word is "believe", which is what turns a prediction about the efforts into an expressed conviction of the holder. The surrounding pronoun belongs to the identification of the holder, and the material about helping the community is the content of the belief rather than the expression of it.

Step 5: What is believed is that the efforts will help the community take steps toward young people developing well — a favourable expectation of them, so the conviction is expressed in support of the efforts. Polarity = Positive.

Step 6: One tuple for the one expressed belief; the factual framing about the year and the leadership was not turned into a second opinion. Each extracted span occurs verbatim in the sentence, punctuation included.""",
        "output": {
            "opinions": [
                {
                    "Source": ["MCCOY , Inc."],
                    "Target": ["two efforts"],
                    "Polar_expression": ["believe"],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """Participants in the Youth Outcomes Educational Initiative identified three goals for all our youth to obtain : economic self - sufficiency , healthy family , social relationships , and involvement in the community .""",
        "reasoning": """Step 1: The sentence reports one act: a group settled on a set of goals, which are then spelled out after the colon. There is a single attitudinal act here, so one opinion.

Step 2: The people who performed it are called "Participants". That noun alone names who they are; the prepositional phrase after it says which programme they took part in, which is circumstantial to identifying the holder. Source = ["Participants"].

Step 3: What they settled on is the substance of the goals, listed after the colon as economic self-sufficiency, a healthy family, social relationships and community involvement. The lead-in words about a count of goals and who they are for do not name any of that substance, and dropping any item from the list would lose part of what was identified. Target = ["economic self - sufficiency , healthy family , social relationships , and involvement in the community"].

Step 4: "identified" is the word that reports the group's mental act of picking out these goals; nothing else in the sentence adds an attitude. It is retained on its own.

Step 5: Reporting that a group identified certain goals says that they named them, without any indication of how favourably or unfavourably they regarded anything. Polarity = Neutral.

Step 6: One tuple for the single act, with all four fields filled. The list Target and the one-word Source and Polar_expression were each checked against the sentence and match exactly, spacing and commas included.""",
        "output": {
            "opinions": [
                {
                    "Source": ["Participants"],
                    "Target": [
                        "economic self - sufficiency , healthy family , social relationships , and involvement in the community"
                    ],
                    "Polar_expression": ["identified"],
                    "Polarity": "Neutral",
                }
            ]
        },
    },
    {
        "text": """As a provider of youth services , MCCOY , Inc. is here to support your valuable efforts to develop young people .""",
        "reasoning": """Step 1: The opening phrase states the organization's role, and the main clause declares a commitment to back what the addressee is doing. That declared commitment is the one attitudinal act in the sentence.

Step 2: The one making the commitment is named in the main clause as "MCCOY , Inc.". The introductory role description characterises that same organization rather than naming a different holder. Source = ["MCCOY , Inc."].

Step 3: What the commitment is directed at is the addressee's work, referred to as "efforts". That noun is enough to pick out what will be backed; the possessive and the purpose clause tell us whose the efforts are and what they aim at. Target = ["efforts"].

Step 4: "support" is the word that states the stance being taken toward those efforts. The adjective in the noun phrase is part of how the addressee's work is referred to in this courteous framing, not the stance the organization is announcing about itself, and the copular "is here to" only says that the organization stands ready.

Step 5: Declaring readiness to support someone's work states an intention to assist rather than passing judgment on the work's quality; it is an alignment of stance without a favourable or unfavourable assessment attached. Polarity = Neutral.

Step 6: One tuple for the single declared commitment; the role-describing opener was not treated as a separate opinion. Every span is an exact substring of the sentence and could not be shortened further.""",
        "output": {
            "opinions": [
                {
                    "Source": ["MCCOY , Inc."],
                    "Target": ["efforts"],
                    "Polar_expression": ["support"],
                    "Polarity": "Neutral",
                }
            ]
        },
    },
    {
        "text": """The US President preferred a more realistic attitude , because it is hard to believe that it will be possible to fully eliminate pollutant gas emissions in one decade .""",
        "reasoning": """Step 1: The main clause reports a preference held by a head of state. The clause introduced by "because" gives the reasoning behind that preference, so it is grounds rather than a further opinion. One opinion.

Step 2: The holder is identified by office as "The US President", which is the only way the sentence names who holds the preference. Source = ["The US President"].

Step 3: What is preferred is the stance described as "a more realistic attitude". The whole phrase is needed, since it is the comparative characterisation of the attitude that makes it the thing chosen over the alternative. Target = ["a more realistic attitude"].

Step 4: "preferred" is the word expressing the stance — it reports a choice in favour of one option over others. The causal clause explains the motivation and is left out, since it states a difficulty about emissions targets rather than the preference itself.

Step 5: To prefer something is to come out in favour of it, so the stance toward the attitude in question is favourable. Polarity = Positive.

Step 6: One tuple, matching the single preference identified at the start, with the justification clause correctly excluded. All spans were verified to appear in the sentence exactly as written.""",
        "output": {
            "opinions": [
                {
                    "Source": ["The US President"],
                    "Target": ["a more realistic attitude"],
                    "Polar_expression": ["preferred"],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """US And UK Criticise Mugabe's Victory""",
        "reasoning": """Step 1: This headline states one thing: two governments have objected to an electoral outcome. There is a single attitudinal act, hence one opinion.

Step 2: The holders are the two states named jointly as the subject, "US And UK". They act together in the one reported act of criticism, so they form a single holder span rather than two. Source = ["US And UK"].

Step 3: The object of the criticism is the electoral outcome, named as "Mugabe's Victory". The possessive is what identifies whose victory is at issue, so it cannot be dropped. Target = ["Mugabe's Victory"].

Step 4: "Criticise" is the word carrying the stance; the headline contains no modifier or hedge around it, so the verb alone is the full expression.

Step 5: To criticise something is to come out against it, and nothing in this compressed headline suggests any other reading. Polarity = Negative.

Step 6: One tuple for the single reported act. All three spans, including the capitalisation used in the headline, match the original exactly.""",
        "output": {
            "opinions": [
                {
                    "Source": ["US And UK"],
                    "Target": ["Mugabe's Victory"],
                    "Polar_expression": ["Criticise"],
                    "Polarity": "Negative",
                }
            ]
        },
    },
    {
        "text": """\" SA wants Mugabe to accommodate the opposition in his new government to form a coalition government , " said an official source .""",
        "reasoning": """Step 1: The sentence consists of a quoted statement plus its attribution. The statement expresses one desire about how a new government should be composed, so one opinion is present.

Step 2: The desire reaches us as a quoted statement, and the sentence identifies who voiced it in the attribution: "an official source". That phrase, vague as it is, is the sentence's explicit designation of the one speaking. Source = ["an official source"].

Step 3: What is wanted is a course of action, and the whole infinitival stretch from "Mugabe" through "coalition government" is that course of action. Trimming it would drop either who must act or what specifically he must do, both of which are part of what is being asked for. Target = ["Mugabe to accommodate the opposition in his new government to form a coalition government"].

Step 4: "wants" is the word expressing the stance — it presents the described arrangement as a desired outcome. The rest of the quotation is the content of the wish, and the attribution verb only says that the wish was voiced.

Step 5: Wanting an outcome puts the holder in favour of it, so the stance toward the proposed coalition arrangement is a supportive one. Polarity = Positive.

Step 6: One tuple for the single expressed desire, with the reporting frame kept out of the evaluative span. Each span was checked against the sentence, quotation marks and spacing included, and matches it exactly.""",
        "output": {
            "opinions": [
                {
                    "Source": ["an official source"],
                    "Target": [
                        "Mugabe to accommodate the opposition in his new government to form a coalition government"
                    ],
                    "Polar_expression": ["wants"],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """International condemnation of Mugabe's win mounted yesterday with US President George Bush and British Foreign Secretary Jack Straw delivering further criticism .""",
        "reasoning": """Step 1: Two distinct attitudinal acts are reported. The first is the broad condemnation whose growth the sentence describes; the second is the specific act of criticism attributed to two named officials in the "with" clause. Because they have different holders, they are two independent opinions rather than one restated.

Step 2: For the first, the only indication of who is condemning is the modifier "International", which places the condemnation with the world community at large; that word is the sentence's sole explicit trace of that holder. For the second, the holders are named in full by office and name, as US President George Bush and British Foreign Secretary Jack Straw, kept as one span since they act jointly in the same reported act. 

Step 3: Both acts are aimed at the same thing, the electoral outcome named as "Mugabe's win". The possessive identifies whose win, so it stays; nothing shorter picks out the event. Target = ["Mugabe's win"] in both tuples.

Step 4: The first act is expressed by the noun "condemnation", which names the stance itself; the reporting verb about it mounting describes how it accumulated rather than what it is. The second is expressed by "delivering further criticism", where "further" matters because it marks this as an addition to the condemnation already under way, and the participle is what states that the two officials issued it.

Step 5: Condemnation and criticism both place their object in an unfavourable light, and the sentence gives no ironic framing. Polarity = Negative for both.

Step 6: Two opinions identified and two tuples produced, distinguished by holder rather than duplicated. All spans, including the long official titles, appear verbatim in the sentence, and none can be shortened without losing part of the identification.""",
        "output": {
            "opinions": [
                {
                    "Source": ["International"],
                    "Target": ["Mugabe's win"],
                    "Polar_expression": ["condemnation"],
                    "Polarity": "Negative",
                },
                {
                    "Source": [
                        "US President George Bush and British Foreign Secretary Jack Straw"
                    ],
                    "Target": ["Mugabe's win"],
                    "Polar_expression": ["delivering further criticism"],
                    "Polarity": "Negative",
                },
            ]
        },
    },
    {
        "text": """He hinted at a tougher policy against Mugabe , saying that the European Union would review sanctions at an upcoming summit meeting in Barcelona , Spain .""",
        "reasoning": """Step 1: The sentence reports that a speaker signalled a harder line toward a named leader, and then relays what he said about a forthcoming review of sanctions. The second part is the reported content supporting the signal, described in procedural terms, so there is one opinion.

Step 2: The holder appears only as the pronoun "He", which is nonetheless an explicit word in the sentence standing for the speaker. Source = ["He"].

Step 3: The person the harder line is directed against is "Mugabe", named at the end of the main clause. The policy is the means by which the stance is taken, not the entity it is taken toward. Target = ["Mugabe"].

Step 4: The stance is carried by "tougher policy against": the comparative is what marks the hardening, and the preposition is what aims that hardening at a particular person, so both are needed. "hinted at" reports only how indirectly the stance was conveyed, and the clause about the sanctions review names a procedural step rather than the stance itself.

Step 5: Signalling a tougher line against someone positions the holder unfavourably toward them, and the reference to reviewing sanctions confirms that direction. Polarity = Negative.

Step 6: One tuple for the single signalled stance; the reported remarks about the summit were not turned into an additional opinion. Every span is an exact, minimal substring of the sentence.""",
        "output": {
            "opinions": [
                {
                    "Source": ["He"],
                    "Target": ["Mugabe"],
                    "Polar_expression": ["tougher policy against"],
                    "Polarity": "Negative",
                }
            ]
        },
    },
],
    "darmstadt_unis": [
    {
        "text": """I'm in my second block of classes , and to be perfectly honest the classes are way way too easy .""",
        "reasoning": """Step 1: The first clause, "I'm in my second block of classes", only situates the writer in a stage of study and carries no evaluation. The evaluative content sits entirely in the second clause, where the classes receive a judgment about their difficulty, so there is one independent opinion here.

Step 2: The pronoun "I" appears, but it belongs to the factual statement about which block of classes the writer is in, and the parenthetical "to be perfectly honest" is a stance marker rather than a named holder attached to the judgment. The evaluation of the classes is stated as a plain property, with nobody named as the one asserting it. Source = [].

Step 3: What is judged is the coursework itself, named as "classes" in the evaluating clause. The article adds nothing to the identification of the entity, and no narrower or more specific alternative appears. Target = ["classes"].

Step 4: The judgment is carried by "too easy" — "too" is what turns a description of low difficulty into a complaint about it, so it cannot be dropped. The doubled "way way" only amplifies an evaluation already complete without it, so it stays out of the span.

Step 5: For a course of study, being easier than it should be is a shortcoming, and "too" marks the difficulty as falling short of what the writer expected. Polarity = Negative.

Step 6: One evaluative clause, one tuple; the opening clause was correctly treated as background rather than a second opinion. Both spans occur in the sentence exactly as written, and neither can be trimmed without losing the complaint.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["classes"],
                    "Polar_expression": ["too easy"],
                    "Polarity": "Negative",
                }
            ]
        },
    },
    {
        "text": """Staff I would rate a 3/5 , they are courteous , but can take a long time to respond .""",
        "reasoning": """Step 1: Three fragments follow the topic word "Staff": a numeric score, an adjectival predication, and a statement about response time. The score "a 3/5" is a figure rather than a sentiment-bearing wording, and "can take a long time to respond" hedges with "can" to report how the staff sometimes behave rather than voicing an attitude toward it. The one fragment that states an evaluation in words is "they are courteous", so one opinion is present.

Step 2: "I" occurs, but it is the subject of the rating clause "I would rate", not of the courteousness claim, which is asserted flatly as a characteristic of the staff. No holder is attached to the opinion being extracted. Source = [].

Step 3: The entity described as courteous is the staff, introduced at the front of the sentence as "Staff"; the later "they" merely points back to it and is not itself informative about who is meant. Target = ["Staff"].

Step 4: The evaluation reduces to the single adjective "courteous". The copula "are" is grammatical support and the pronoun belongs to the Target's reference chain, so neither adds evaluative content.

Step 5: Describing staff as courteous credits them with good manners toward the people they deal with, and nothing in the wording undercuts that reading. Polarity = Positive.

Step 6: One tuple, matching the single fragment that states an evaluation in evaluative words. The numeric score and the hedged remark about response time were not converted into extra opinions. Both spans match the sentence character-for-character.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["Staff"],
                    "Polar_expression": ["courteous"],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """The field reps ( some-most ) are rude and don't want to do their jobs .""",
        "reasoning": """Step 1: One complaint is made about the field reps, and the coordinated clause "don't want to do their jobs" spells out how that complaint shows itself in their behaviour rather than raising a separate matter. The parenthetical "( some-most )" only qualifies how many reps are meant. This is a single opinion.

Step 2: Nobody is named as the person making the charge; the sentence states it directly as a fact about the reps. Source = [].

Step 3: The people criticised are the "field reps". The determiner "The" and the parenthetical quantifier are not needed to identify who is meant, so the name of the role alone suffices. Target = ["field reps"].

Step 4: "rude" is the word that delivers the criticism; the copula and the coordinated behavioural clause elaborate on it without adding a distinct evaluative term. The adjective on its own preserves the full force of the charge.

Step 5: Calling people who serve students rude is a straightforward accusation of poor treatment. Polarity = Negative.

Step 6: One opinion identified in Step 1 and one tuple produced. The elaborating clause was kept out rather than treated as a second evaluation. The extracted spans appear verbatim in the sentence and cannot be shortened.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["field reps"],
                    "Polar_expression": ["rude"],
                    "Polarity": "Negative",
                }
            ]
        },
    },
    {
        "text": """That can be rather depressing .""",
        "reasoning": """Step 1: The whole sentence is a single predication assigning an emotional effect to something referred to back as "That". There are no further clauses to consider, so exactly one opinion is present.

Step 2: The sentence has no first-person pronoun and names no speaker; the emotional effect is asserted without attribution. Source = [].

Step 3: The evaluated thing is whatever "That" points back to — the sentence itself offers only the demonstrative as the name for it, and a demonstrative standing in for a described situation is still the explicit mention of what is being judged. Target = ["That"].

Step 4: The evaluative material consists of the adjective "depressing", which carries the effect being reported, together with the degree word "rather", which states how strongly the effect is felt and so is part of the assessment. The modal copula "can be" is grammatical framing about how often this holds, so it stays outside the span.

Step 5: Describing something as depressing reports that it lowers one's mood, and the hedging by degree softens the strength without changing its direction. Polarity = Negative.

Step 6: A single predication yields a single tuple, with all four fields filled. Each extracted piece occurs in the sentence exactly as written, and dropping either the adjective or the degree word would lose part of what is asserted.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["That"],
                    "Polar_expression": ["rather", "depressing"],
                    "Polarity": "Negative",
                }
            ]
        },
    },
    {
        "text": """Don't even bother emailing the administration .""",
        "reasoning": """Step 1: The sentence is one piece of advice telling the reader that a particular attempt at contact is pointless. There is a single evaluative move here, so one opinion.

Step 2: The imperative addresses the reader and leaves the adviser unnamed; no word in the sentence identifies who is speaking. Source = [].

Step 3: The body whose responsiveness is being dismissed is the "administration". The determiner is not needed to identify it, and "emailing" names the action through which the complaint is made rather than the entity judged. Target = ["administration"].

Step 4: "Don't even bother" is what conveys the judgment: "even" makes the point that the attempt is not merely unlikely to work but not worth starting. Trimming any part of the phrase would leave either a bare negation or a verb without its dismissive force.

Step 5: Telling the reader that writing to the administration is not worth the effort asserts that it will not answer usefully, which is a complaint about it. Polarity = Negative.

Step 6: One tuple for one piece of advice, all fields present. The spans are exact substrings of the sentence, and no consequence or explanation was mistaken for an additional opinion.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["administration"],
                    "Polar_expression": ["Don't even bother"],
                    "Polarity": "Negative",
                }
            ]
        },
    },
    {
        "text": """UMUC brags how they cut breaks financially for military .""",
        "reasoning": """Step 1: The sentence reports what the institution says about its own financial concessions, and the choice of reporting verb is itself the evaluative act. The content of the claim about breaks for military students is what is being reported, not a second opinion. One opinion.

Step 2: UMUC is the one doing the boasting, but the attitude captured in this sentence is the writer's characterisation of that boasting, and no word identifies the writer. Source = [].

Step 3: The institution whose conduct is being characterised is "UMUC", named directly at the head of the sentence. The reported content is what it boasts about rather than the thing evaluated. Target = ["UMUC"].

Step 4: "brags" is the single word doing the evaluating — a neutral report would have used "says" or "advertises", and this choice ascribes self-congratulation. Nothing outside that verb is needed, since the following clause only supplies the subject matter of the boast.

Step 5: Saying that an institution brags accuses it of trumpeting its own generosity, which is a criticism of how it presents itself. Polarity = Negative.

Step 6: One opinion, one tuple, four fields filled. The reported clause was left out rather than promoted to a separate evaluation, and both spans occur verbatim in the sentence.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["UMUC"],
                    "Polar_expression": ["brags"],
                    "Polarity": "Negative",
                }
            ]
        },
    },
    {
        "text": """UMUC should be ashamed of themselves for even calling themselves a university .""",
        "reasoning": """Step 1: One condemnation is made, followed by the grounds for it in the "for even calling..." clause. The grounds explain why the condemnation is deserved and are not an independent evaluation, so there is one opinion.

Step 2: The verdict is delivered without anyone claiming it; there is no pronoun or name marking who holds it. Source = [].

Step 3: The institution being condemned is "UMUC", the subject of the sentence. Target = ["UMUC"].

Step 4: "should be ashamed" carries the verdict — the modal is what makes it a judgment on the institution rather than a report of its feelings. The reflexive "of themselves" adds no evaluative content beyond the phrase, and the causal clause states the reason for the verdict rather than the verdict itself.

Step 5: Telling an institution it ought to feel shame is about as direct a condemnation as the language allows, with no ironic reversal in play. Polarity = Negative.

Step 6: One tuple matching the single condemnation from Step 1, with the explanatory clause correctly excluded. Every span is an exact, minimal substring of the sentence.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["UMUC"],
                    "Polar_expression": ["should be ashamed"],
                    "Polarity": "Negative",
                }
            ]
        },
    },
    {
        "text": """PLEASE , do not enroll in the University of Phoenix Online .""",
        "reasoning": """Step 1: The sentence is one warning against enrolling at a named institution. "PLEASE" intensifies the appeal but states nothing on its own, so one opinion is present.

Step 2: The warning is addressed to the reader with no mention of who is issuing it. Source = [].

Step 3: The institution the reader is warned away from is named as the "University of Phoenix"; that is the span that identifies the school being judged, and the surrounding determiner is not part of identifying it. Target = ["University of Phoenix"].

Step 4: "do not enroll" is the evaluative core: advising against enrolment is how the negative assessment is expressed here. The capitalised appeal "PLEASE" only signals urgency and is not part of the assessment.

Step 5: Urging prospective students not to enrol implies the institution is not worth attending, so the intent of the warning is critical. Polarity = Negative.

Step 6: One warning, one tuple, all four fields filled. The extracted spans match the sentence exactly, and the emphatic appeal was not misread as a second opinion.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["University of Phoenix"],
                    "Polar_expression": ["do not enroll"],
                    "Polarity": "Negative",
                }
            ]
        },
    },
    {
        "text": """Sucking you into enrollment is just about the ONLY thing this school is good at .""",
        "reasoning": """Step 1: The sentence makes one claim: that a particular activity is the school's one area of competence. The subject phrase "Sucking you into enrollment" names that activity; it is what the competence is about, not a separate evaluation. One opinion.

Step 2: The claim is asserted without attribution — "you" is the addressee, not the holder, and no speaker is named. Source = [].

Step 3: The entity whose competence is at issue is "this school", named explicitly in the predicate. Target = ["this school"].

Step 4: The evaluation is built from "ONLY thing", which restricts the competence to a single area and is capitalised to press that restriction, and "good", which states the competence itself; the words in between name the school and so belong to the Target, leaving the span in two pieces. Neither piece can go: without "good" there is no ascribed ability, and without "ONLY thing" the restriction is lost.

Step 5: Read against its immediate Target, the predicate ascribes to the school an ability it does possess — it is credited with being good at the named activity — so the local evaluation of the school in this predication is favourable, whatever the barbed effect of the sentence as a whole. Polarity = Positive.

Step 6: One tuple for the single claim identified at the start. The subject phrase was treated as the domain of the competence rather than as another opinion. Both fragments of the evaluative span and the Target occur verbatim in the sentence.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["this school"],
                    "Polar_expression": ["ONLY thing", "good"],
                    "Polarity": "Positive",
                }
            ]
        },
    },
    {
        "text": """I will tell you firsthand this school DOES not care about your education...only the money you give them .""",
        "reasoning": """Step 1: "I will tell you firsthand" announces that a claim is coming and vouches for it, without evaluating anything itself. The claim that follows is a single charge about what the school does and does not care about, with the part after the ellipsis completing the same contrast rather than opening a new one. One opinion.

Step 2: "I" appears, but only as the subject of the announcing frame "I will tell you"; the charge against the school is stated as a fact about the school and not attributed to a named holder within it. Source = [].

Step 3: The institution charged is picked out by "school"; the demonstrative merely locates which one is meant and adds nothing to naming the entity. Target = ["school"].

Step 4: What conveys the charge is the denial of concern: "not", which supplies the absence, and "care", the attitude being denied. The capitalised "DOES" only adds emphasis to that denial, and "about your education" names what the school is indifferent to rather than expressing the indifference.

Step 5: Saying that a school does not care about its students' education denies it the very concern its purpose requires, and the trailing contrast about money reinforces that reading. Polarity = Negative.

Step 6: One tuple for the single charge, with the announcing frame and the money contrast both excluded as scaffolding and reinforcement. All extracted pieces appear in the sentence exactly as written, and no field is empty that should be filled.""",
        "output": {
            "opinions": [
                {
                    "Source": [],
                    "Target": ["school"],
                    "Polar_expression": ["not", "care"],
                    "Polarity": "Negative",
                }
            ]
        },
    },
]
}


def load_examples_pool(path: Optional[str], dataset: str) -> list[dict[str, Any]]:
    if path:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Examples pool JSON must be a list, got: {type(data)}")
        return data

    # Lấy dữ liệu tĩnh từ dataset yêu cầu
    pool = _HARDCODED_POOL.get(dataset, [])
    if pool:
        return list(pool)

    # Fallback 1: dataset khác cùng ngôn ngữ (nếu có pool không rỗng).
    language = DATASET_LANGUAGES.get(dataset)
    if language is not None:
        for other_dataset, other_language in DATASET_LANGUAGES.items():
            if other_language == language and _HARDCODED_POOL.get(other_dataset):
                print(
                    f"Warning: Hardcoded CoT pool for dataset '{dataset}' is empty. "
                    f"Falling back to sibling dataset '{other_dataset}' (same language: '{language}')."
                )
                return list(_HARDCODED_POOL[other_dataset])

    # Fallback 2: cuối cùng mới dùng opener_en (giữ đúng tinh thần fallback cũ
    # "khác 'en' thì fallback về 'en'", chỉ khác là giờ trỏ đích danh 1 dataset).
    if dataset != "opener_en" and _HARDCODED_POOL.get("opener_en"):
        print(
            f"Warning: No CoT pool found for dataset '{dataset}' or any sibling "
            f"dataset in the same language. Falling back to 'opener_en'."
        )
        return list(_HARDCODED_POOL["opener_en"])

    return []

