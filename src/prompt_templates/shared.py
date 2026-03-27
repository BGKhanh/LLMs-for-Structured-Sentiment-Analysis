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
    # NOTE: content copied from legacy `FewShotPrompt._get_system_prompt_vi/en`.
    "vi": """Bạn là chuyên gia trong lĩnh vực phân tích cảm xúc tiếng Việt có cấu trúc. 
Nhiệm vụ của bạn là phân tích bình luận mạng xã hội và trích xuất các thành phần cảm xúc theo cấu trúc JSON.

ĐỊNH NGHĨA CÁC THÀNH PHẦN:

1. SOURCE (Nguồn gốc bình luận):
   - Người phát biểu ý kiến, có thể là người bình luận hoặc được trích dẫn
   - Thường là các đại từ nhân xưng: "Tôi", "Tao", "Mình", "Bọn tao", "Mẹ tui"
   - Có thể có hoặc không có trong câu

2. TARGET (Đối tượng hướng tới):
   - Cá nhân, tập thể, sự vật, hiện tượng mà bình luận hướng đến
   - Thường là các đại từ xưng hô: "Mày", "Cậu", "Anh ấy", "Bạn"
   - Có thể có hoặc không có trong câu

3. POLAR_EXPRESSION (Biểu thức cảm xúc):
   - Từ/cụm từ bày tỏ cảm xúc, ý nghĩ, cảm nhận, hành động
   - Bao gồm: tính từ cảm xúc, thán từ, hành động xúc phạm/khen ngợi
   - Ví dụ: "buồn", "vui", "tức giận", "đáng đời", "đánh"
   - BẮT BUỘC phải có

4. POLARITY (Tính chất cảm xúc):
   - Positive: Khích lệ, động viên, chia sẻ, vui đùa không xúc phạm
   - Negative: Xúc phạm, kích động, chia rẽ, gây thù ghét
   - Neutral: Bình luận bình thường, khách quan

5. INTENSITY (Cường độ cảm xúc):
   - Strong: Cảm xúc mạnh mẽ, từ ngữ quyết liệt
   - Standard: Cảm xúc bình thường, từ ngữ thông thường  
   - Weak: Cảm xúc nhẹ nhàng, từ ngữ dè dặt

QUY TẮC PHÂN TÍCH:
- Mỗi câu có thể chứa nhiều opinion khác nhau. 
- Mỗi opinion phải có ít nhất 1 Polar_expression
- Polar_expression là thành phần bắt buộc phải có
- Chú ý các từ viết tắt, teencode, hàm ý, ẩn ý trong tiếng Việt

FORMAT JSON OUTPUT:
{
  "sent_id": "[ID của câu]",
  "text": "[Bình luận gốc]",
  "opinions": [
    {
      "Source": ["text_span_1"],
      "Target": ["text_span_1"],
      "Polar_expression": ["text_span_1"],
      "Polarity": "Positive/Negative/Neutral",
      "Intensity": "Strong/Standard/Weak"
    }
  ]
}

""",
    "en": """You are an expert in structured Vietnamese sentiment analysis. 
Your task is to analyze social media comments and extract sentiment components in a JSON structure.

DEFINITIONS OF COMPONENTS:

1. SOURCE:
    - The speaker of the opinion, who can be the commenter or someone quoted.
    - Often personal pronouns: "Tôi" (I), "Tao" (I/me, informal), "Mình" (I/me/we, inclusive), "Bọn tao" (We, informal), "Mẹ tui" (My mother)
    - May or may not be present in the sentence.

2. TARGET:
    - The individual, group, object, or phenomenon that the comment is directed towards.
    - Often second-person pronouns: "Mày" (You, informal), "Cậu" (You, polite), "Anh ấy" (He/Him), "Bạn" (You)
    - May or may not be present in the sentence.

3. POLAR_EXPRESSION:
    - Word(s)/phrase(s) that express emotion, thought, feeling, or action.
    - Includes: emotional adjectives, interjections, insulting/praising actions.
    - Examples: "buồn" (sad), "vui" (happy), "tức giận" (angry), "đáng đời" (serves you right), "đánh" (to hit/beat)
    - MUST be present.

4. POLARITY:
    - Positive: Encouragement, motivation, sharing, non-offensive joking.
    - Negative: Insulting, inciting, divisive, causing hatred.
    - Neutral: Normal, objective comment.

5. INTENSITY:
    - Strong: Intense emotion, strong language.
    - Standard: Normal emotion, common language.  
    - Weak: Mild emotion, reserved language.

ANALYSIS RULES:
- Each sentence may contain multiple different opinions. 
- Each opinion must have at least 1 Polar_expression.
- Polar_expression is a required component.
- Pay attention to abbreviations, teencode (Vietnamese slang), implications, and hidden meanings in Vietnamese.

JSON OUTPUT FORMAT:
{
  "sent_id": "[Sentence ID]",
  "text": "[Original comment]",
  "opinions": [
    {
      "Source": ["text_span_1"],
      "Target": ["text_span_1"],
      "Polar_expression": ["text_span_1"],
      "Polarity": "Positive/Negative/Neutral",
      "Intensity": "Strong/Standard/Weak"
    }
  ]
}

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
            f"Unsupported language: '{language}'. Available: {sorted(SYSTEM_PROMPTS.keys())}"
        )
    return SYSTEM_PROMPTS[language]


# ---------------------------------------------------------------------------
# Hardcoded pools (fallback)
# ---------------------------------------------------------------------------
# NOTE: copied from legacy `Re2PaSCoTPrompt.EXAMPLES_POOL_VI/EN` (current state).
_HARDCODED_POOL: dict[str, list[dict[str, Any]]] = {
    "vi": [
        {
            "text": "quay quay cái lồn , thấy bị bắt nạt thì ra nói một câu bảo vệ người ta , có khi tối về lại có người nằm ôm . cứ cầm điện thoại pin với chả không pin",
            "reasoning": """
Bước 1: Trích xuất Ứng viên (Extraction)
Tôi tìm thấy 5 cụm từ mang sắc thái cảm xúc hoặc mô tả hành vi: 
1. "quay quay cái lồn"
2. "thấy bị bắt nạt"
3. "nói một câu bảo vệ người ta"
4. "tối về lại có người nằm ôm"
5. "cứ cầm điện thoại pin với chả không pin"

Bước 2: Lập Kế hoạch (Planning)
- Rà soát ứng viên: Cả 5 cụm từ đều mang sắc thái cảm xúc hoặc mô tả hành vi rõ ràng, không có cụm từ nào cần loại bỏ.
- Chiến lược xử lý:
  + Thứ tự: Phân tích tuần tự theo văn bản.
  + Source/Target ẩn: Vì câu thiếu chủ ngữ, tôi sẽ suy luận Source là người nói (tác giả) và Target dựa trên hành động.
- Nhận diện thách thức:
  + Văn bản chứa từ ngữ thô tục ("cái lồn") và tiếng lóng, cần xác định xem đây là xúc phạm hay chỉ là than phiền.
  + Cấu trúc khẩu ngữ "A với chả B" ("pin với chả không pin") là một thách thức về thái độ, thường mang nghĩa dè bỉu.
- Xác nhận: Tôi sẽ phân tích đầy đủ 5 thành phần (Source, Target, Polar_expression, Polarity, Intensity) cho mỗi Opinion.

Bước 3: Thực thi Suy luận (Execution)
--- Opinion 1 ---
- Source: Không có (ẩn chủ ngữ).
- Target: Không có (ẩn đối tượng bị chửi).
- Polar_expression: "quay quay cái lồn" -> Đây là cụm từ văng tục (tiếng lóng) thể hiện sự bực bội.
- Polarity: Negative -> Do sử dụng từ ngữ xúc phạm ("cái lồn").
- Intensity: Standard -> Mức độ thô tục rõ ràng nhưng phổ biến trong văn nói.

--- Opinion 2 ---
- Source: Không có.
- Target: Không có.
- Polar_expression: "thấy bị bắt nạt" -> Mô tả một sự việc tiêu cực.
- Polarity: Negative -> "Bắt nạt" mang nghĩa xấu.
- Intensity: Standard.

--- Opinion 3 ---
- Source: Không có.
- Target: Không có ("người ta" là đại từ phiếm chỉ).
- Polar_expression: "nói một câu bảo vệ người ta" -> Hành động bênh vực.
- Polarity: Positive -> "Bảo vệ" là hành động tích cực, trượng nghĩa.
- Intensity: Standard.

--- Opinion 4 ---
- Source: Không có.
- Target: Không có.
- Polar_expression: "tối về lại có người nằm ôm" -> Ám chỉ sự an ủi, tình cảm.
- Polarity: Positive -> Thể hiện sự ấm áp, kết cục tốt.
- Intensity: Standard.

--- Opinion 5 ---
- Source: Không có.
- Target: Không có.
- Polar_expression: "cứ cầm điện thoại pin với chả không pin" -> Cấu trúc "A với chả B" thường dùng để phàn nàn.
- Polarity: Neutral -> Chỉ nhận xét hành vi mà không có tính xúc phạm hay khen ngợi rõ ràng.
- Intensity: Standard.""",
            "output": {
                "sent_id": 1833,
                "text": "quay quay cái lồn , thấy bị bắt nạt thì ra nói một câu bảo vệ người ta , có khi tối về lại có người nằm ôm . cứ cầm điện thoại pin với chả không pin",
                "opinions": [
                    {
                        "Source": [],
                        "Target": [],
                        "Polar_expression": ["quay quay cái lồn"],
                        "Polarity": "Negative",
                        "Intensity": "Standard",
                    },
                    {
                        "Source": [],
                        "Target": [],
                        "Polar_expression": ["thấy bị bắt nạt"],
                        "Polarity": "Negative",
                        "Intensity": "Standard",
                    },
                    {
                        "Source": [],
                        "Target": [],
                        "Polar_expression": ["nói một câu bảo vệ người ta"],
                        "Polarity": "Positive",
                        "Intensity": "Standard",
                    },
                    {
                        "Source": [],
                        "Target": [],
                        "Polar_expression": ["tối về lại có người nằm ôm"],
                        "Polarity": "Positive",
                        "Intensity": "Standard",
                    },
                    {
                        "Source": [],
                        "Target": [],
                        "Polar_expression": ["cứ cầm điện thoại pin với chả không pin"],
                        "Polarity": "Neutral",
                        "Intensity": "Standard",
                    },
                ],
            },
        },
        {
            "text": "yêu sắp 2 năm rồi mà không biết người yêu kể chuyện cho nghe là gì 😢 đòi mấy lần toàn kêu không bình thường xịu xịu",
            "reasoning": """Bước 1: Trích xuất Ứng viên (Extraction)
Tôi tìm thấy 3 cụm từ mang sắc thái cảm xúc hoặc mô tả trạng thái:
1. "yêu sắp 2 năm rồi"
2. "không biết người yêu kể chuyện cho nghe là gì"
3. "đòi mấy lần toàn kêu không bình thường xịu xịu"

Bước 2: Lập Kế hoạch (Planning)
- Rà soát ứng viên: Tất cả ứng viên đều hợp lệ.
- Chiến lược xử lý:
  + Thứ tự: Phân tích tuần tự.
  + Cách tiếp cận: Tập trung giải mã các tín hiệu phi ngôn ngữ (emoji) và sắc thái từ vựng để xác định Polarity.
  + Source/Target ẩn: Vì câu thiếu chủ ngữ, tôi sẽ suy luận Source là người nói (tác giả) và Target dựa trên hành động.
- Nhận diện thách thức:
  + Thách thức về ngữ nghĩa thời gian: Cụm "sắp 2 năm" cần được xác định là thành tựu (tích cực) hay sự kéo dài lê thê (tiêu cực) dựa trên ngữ cảnh "yêu".
  + Thách thức về đa phương thức: Cần kết hợp emoji "😢" với nội dung văn bản để chốt Polarity Negative.
  + Thách thức về từ láy: Từ "xịu xịu" có sắc thái nhẹ, cần cân nhắc kỹ giữa Neutral và Negative.
- Xác nhận: Tôi sẽ phân tích đầy đủ 5 thành phần cho 3 Opinion trên.

Bước 3: Thực thi Suy luận (Execution)
--- Opinion 1 ---
- Source: Không có (ẩn chủ ngữ).
- Target: Không có.
- Polar_expression: "yêu sắp 2 năm rồi" -> Thể hiện mối quan hệ tình cảm lâu dài, bền vững.
- Polarity: Positive -> Thời gian dài "2 năm" trong tình yêu thường mang ý nghĩa tích cực, gắn bó.
- Intensity: Standard -> Diễn đạt mức độ bình thường.

--- Opinion 2 ---
- Source: Không có.
- Target: Không có ("người yêu" là đối tượng được nhắc đến nhưng ngữ cảnh là sự than phiền về hành động, không nhắm trực tiếp vào con người).
- Polar_expression: "không biết người yêu kể chuyện cho nghe là gì" -> Bày tỏ sự thiếu vắng giao tiếp, cảm giác thiệt thòi (kèm emoji buồn 😢).
- Polarity: Negative -> Thể hiện sự thất vọng, buồn bã vì thiếu kết nối.
- Intensity: Standard -> Mức độ than phiền thông thường.

--- Opinion 3 ---
- Source: Không có.
- Target: Không có.
- Polar_expression: "đòi mấy lần toàn kêu không bình thường xịu xịu" -> Mô tả phản ứng từ chối lặp lại của đối phương ("đòi mấy lần").
- Polarity: Neutral -> "Xịu xịu" có thể hiểu là nhạt nhẽo hoặc buồn, nhưng trong ngữ cảnh này chủ yếu là kể lại sự việc một cách khách quan, chưa đủ gay gắt để thành Negative hay vui vẻ để thành Positive.
- Intensity: Standard.""",
            "output": {
                "sent_id": 1834,
                "text": "yêu sắp 2 năm rồi mà không biết người yêu kể chuyện cho nghe là gì 😢 đòi mấy lần toàn kêu không bình thường xịu xịu",
                "opinions": [
                    {
                        "Source": [],
                        "Target": [],
                        "Polar_expression": ["yêu sắp 2 năm rồi"],
                        "Polarity": "Positive",
                        "Intensity": "Standard",
                    },
                    {
                        "Source": [],
                        "Target": [],
                        "Polar_expression": ["không biết người yêu kể chuyện cho nghe là gì"],
                        "Polarity": "Negative",
                        "Intensity": "Standard",
                    },
                    {
                        "Source": [],
                        "Target": [],
                        "Polar_expression": ["đòi mấy lần toàn kêu không bình thường xịu xịu"],
                        "Polarity": "Neutral",
                        "Intensity": "Standard",
                    },
                ],
            },
        },
        {
            "text": "hi vọng câu chuyện admin vừa bịa ra giúp các bạn có thêm niềm tin trong cuộc sống 😂.",
            "reasoning": """
Bước 1: Trích xuất Ứng viên (Extraction)
Tôi tìm thấy 1 cụm từ mang sắc thái cảm xúc:
1. "giúp các bạn có thêm niềm tin trong cuộc sống"

Bước 2: Lập Kế hoạch (Planning)
- Rà soát ứng viên: Cụm từ duy nhất này hợp lệ và mang nghĩa tích cực rõ ràng.
- Chiến lược xử lý:
  + Truy vết Target: Cần xác định đối tượng gây ra tác động "giúp đỡ" (Cause of Emotion) dù chủ ngữ bị ẩn.
- Nhận diện thách thức:
  + Thách thức về Châm biếm/Hài hước (Sarcasm/Irony): Có sự xung đột giữa nội dung ngữ nghĩa tích cực ("thêm niềm tin") và các tín hiệu giả định/đùa cợt ("bịa ra", emoji "😂").
  + Chiến lược giải quyết: Áp dụng quy tắc "Tách biệt Ngữ nghĩa" (Semantic Disentanglement). Ta sẽ ưu tiên gán nhãn dựa trên Giá trị Cảm xúc Hiển ngôn (Explicit Sentiment) của bản thân cụm từ polar expression, thay vì Ý định Giao tiếp (Pragmatic Intent) của toàn bộ câu.
  + Source/Target ẩn: Vì câu thiếu chủ ngữ, tôi sẽ suy luận Source là người nói (tác giả) và Target dựa trên hành động (ví dụ: hành động quay phim).
- Xác nhận: Tôi sẽ phân tích đầy đủ 5 thành phần cho Opinion này.

Bước 3: Thực thi Suy luận (Execution)
--- Opinion 1 ---
- Source: Không có (ẩn chủ ngữ).
- Target: "câu chuyện admin vừa bịa ra" -> Đây là đối tượng tạo ra tác động, nguyên nhân của cảm xúc.
- Polar_expression: "giúp các bạn có thêm niềm tin trong cuộc sống" -> Thể hiện tác động tích cực, mang lại giá trị tinh thần.
- Polarity: Positive -> "Có thêm niềm tin" và "giúp đỡ" mang ý nghĩa tích cực rõ ràng. Mặc dù emoji 😂 có thể gợi ý sự hài hước hoặc châm biếm ("câu chuyện bịa"), nhưng xét riêng polar expression thì nó vẫn mô tả một kết quả tốt đẹp.
- Intensity: Standard -> Diễn đạt ở mức độ bình thường.""",
            "output": {
                "sent_id": 1835,
                "text": "hi vọng câu chuyện admin vừa bịa ra giúp các bạn có thêm niềm tin trong cuộc sống 😂.",
                "opinions": [
                    {
                        "Source": [],
                        "Target": ["câu chuyện admin vừa bịa ra"],
                        "Polar_expression": ["giúp các bạn có thêm niềm tin trong cuộc sống"],
                        "Polarity": "Positive",
                        "Intensity": "Standard",
                    }
                ],
            },
        },
    ],
    "en": [],
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

