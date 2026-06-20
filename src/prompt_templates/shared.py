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

Bước 5: Phân loại Polarity và gán Intensity

Tuple A — "bộ óc sáng tạo cùng với sự lãnh đạo tài giỏi": từ ngữ "sáng tạo", "tài giỏi" mang tính khen ngợi năng lực → Positive.
Tuple B — "lòng tham hơn người và trí tuệ ngắn hạn": "lòng tham", "ngắn hạn" mang tính phê phán phẩm chất xấu → Negative.
Tuple C — "sẽ dẫn dắt trung nguyên xuống con mẹ nó giếng luôn": thành ngữ ẩn dụ chỉ kết cục tồi tệ/sụp đổ, kèm ngôn từ thô tục nhấn mạnh mức độ tiêu cực → Negative.

Theo Mục E, Intensity cố định = "Standard" cho tất cả các tuple, bất kể mức độ ngôn từ mạnh hay nhẹ.
""",
            "output": """
{
  "opinions": [
    {
      "Source": [],
      "Target": ["bà thảo"],
      "Polar_expression": ["bộ óc sáng tạo cùng với sự lãnh đạo tài giỏi"],
      "Polarity": "Positive",
      "Intensity": "Standard"
    },
    {
      "Source": [],
      "Target": ["bà thảo"],
      "Polar_expression": ["lòng tham hơn người và trí tuệ ngắn hạn"],
      "Polarity": "Negative",
      "Intensity": "Standard"
    },
    {
      "Source": [],
      "Target": ["bà thảo"],
      "Polar_expression": ["sẽ dẫn dắt trung nguyên xuống con mẹ nó giếng luôn"],
      "Polarity": "Negative",
      "Intensity": "Standard"
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

Bước 5: Phân loại Polarity và gán Intensity

Tuple 1 — "dạy cách chống tham nhũng rất rễ làm": từ "rất rễ (dễ) làm" mang sắc thái khẳng định khả năng/sự dễ dàng, một đặc điểm được nêu ra theo hướng tích cực (dễ làm = có lợi, khả thi) → Positive.
Tuple 2 — "có dám làm không": đây là câu hỏi nghi vấn/tu từ, không mang sắc thái yêu/ghét rõ rệt, chỉ chất vấn về việc có thực hiện hay không → theo Mục D (câu hỏi tu từ, câu hỏi nghi vấn → Neutral) → Neutral.

Theo Mục E, Intensity cố định = "Standard" cho cả 2 tuple.
""",
            "output": """
{
  "opinions": [
    {
      "Source": ["anh"],
      "Target": ["Anh em"],
      "Polar_expression": ["dạy cách chống tham nhũng rất rễ làm"],
      "Polarity": "Positive",
      "Intensity": "Standard"
    },
    {
      "Source": [],
      "Target": [],
      "Polar_expression": ["có dám làm không"],
      "Polarity": "Neutral",
      "Intensity": "Standard"
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

Bước 5: Phân loại Polarity và gán Intensity

"hay quá": từ "hay" mang nghĩa khen ngợi, đánh giá tích cực về chất lượng nội dung, kết hợp "quá" nhấn mạnh mức độ khen → Positive.

Theo Mục E, Intensity cố định = "Standard", bất kể từ "quá" tạo cảm giác nhấn mạnh.
""",
            "output": """
{
  "opinions": [
    {
      "Source": [],
      "Target": ["bài viết"],
      "Polar_expression": ["hay quá"],
      "Polarity": "Positive",
      "Intensity": "Standard"
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

Bước 5: Phân loại Polarity và gán Intensity

Tuple 1 — "chốg mắt lên xem": hành động mang tính đe dọa, thách thức, quan sát dò xét với ý đồ tiêu cực (chờ xem đối phương thất bại/lộ bản chất) → Negative.
Tuple 2 — "mặn nồg đến bao giờ": câu hỏi tu từ mang tính mỉa mai, châm biếm, ngầm ý nghi ngờ/chê bai tình cảm của đối tượng sẽ không bền lâu → Negative (không xếp Neutral vì có sắc thái châm biếm/khinh thường rõ rệt, khác với câu hỏi khách quan thuần túy).

Theo Mục E, Intensity cố định = "Standard" cho cả 2 tuple.
""",
            "output": """
{
  "opinions": [
    {
      "Source": ["t"],
      "Target": ["chúg mày"],
      "Polar_expression": ["chốg mắt lên xem"],
      "Polarity": "Negative",
      "Intensity": "Standard"
    },
    {
      "Source": [],
      "Target": ["chúg mày"],
      "Polar_expression": ["mặn nồg đến bao giờ"],
      "Polarity": "Negative",
      "Intensity": "Standard"
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

Bước 5: Phân loại Polarity và gán Intensity
"đã thấy chóng mặt rồi" là một mô tả mang tính thực tế khách quan về phản ứng sinh lý khi nhìn xuống từ độ cao (hiện tượng chóng mặt do độ cao), không phải một lời khen/chê hay biểu lộ cảm xúc yêu/ghét rõ rệt đối với đối tượng nào. Theo Mục D, các phát biểu mô tả trạng thái thực tế khách quan thuộc nhóm Neutral.
→ Polarity = Neutral.
Theo Mục E, Intensity cố định = "Standard".
""",
            "output": """
{
  "opinions": [
    {
      "Source": ["mình"],
      "Target": ["nhìn từ trên cao xuống"],
      "Polar_expression": ["đã thấy chóng mặt rồi"],
      "Polarity": "Neutral",
      "Intensity": "Standard"
    }
  ]
}
"""
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

