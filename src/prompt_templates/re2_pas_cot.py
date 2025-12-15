# src/prompt_templates/re2_pas_cot.py

from typing import Tuple, List, Dict, Any, Optional
import json
import random
from .base import BasePromptTemplate


class Re2PaSCoTPrompt(BasePromptTemplate):
    """
    Combination of Re-reading (Re2) + Plan-and-Solve (PaS) + Chain-of-Thought (CoT).
    
    Structure:
    1. PaS Instruction (System/User start)
    2. CoT Examples (Few-shot with PaS reasoning style)
    3. Input Question
    4. Re-reading Instruction (Repeat Input)
    5. Final Trigger
    """
    
    # Example pool with PaS-style reasoning
    # Placeholder for more examples
    EXAMPLES_POOL_VI = [
        {
            "text": "quay quay cái lồn , thấy bị bắt nạt thì ra nói một câu bảo vệ người ta , có khi tối về lại có người nằm ôm . cứ cầm điện thoại pin với chả không pin",
            "reasoning": """Bước 1: Trích xuất Ứng viên (Extraction)
Tôi tìm thấy 5 cụm từ mang sắc thái cảm xúc hoặc mô tả hành vi: 
1. "quay quay cái lồn"
2. "thấy bị bắt nạt"
3. "nói một câu bảo vệ người ta"
4. "tối về lại có người nằm ôm"
5. "cứ cầm điện thoại pin với chả không pin"

Bước 2: Lập Kế hoạch (Planning)
Tôi sẽ phân tích tuần tự 5 cụm từ trên tương ứng với 5 Opinion riêng biệt để xác định các thành phần cấu trúc của chúng.

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
                        "Intensity": "Standard"
                    },
                    {
                        "Source": [],
                        "Target": [],
                        "Polar_expression": ["thấy bị bắt nạt"],
                        "Polarity": "Negative",
                        "Intensity": "Standard"
                    },
                    {
                        "Source": [],
                        "Target": [],
                        "Polar_expression": ["nói một câu bảo vệ người ta"],
                        "Polarity": "Positive",
                        "Intensity": "Standard"
                    },
                    {
                        "Source": [],
                        "Target": [],
                        "Polar_expression": ["tối về lại có người nằm ôm"],
                        "Polarity": "Positive",
                        "Intensity": "Standard"
                    },
                    {
                        "Source": [],
                        "Target": [],
                        "Polar_expression": ["cứ cầm điện thoại pin với chả không pin"],
                        "Polarity": "Neutral",
                        "Intensity": "Standard"
                    }
                ]
            }
        },
        {
            "text": "yêu sắp 2 năm rồi mà không biết người yêu kể chuyện cho nghe là gì 😢 đòi mấy lần toàn kêu không bình thường xịu xịu",
            "reasoning": """Bước 1: Trích xuất Ứng viên (Extraction)
Tôi tìm thấy 3 cụm từ mang sắc thái cảm xúc hoặc mô tả trạng thái:
1. "yêu sắp 2 năm rồi"
2. "không biết người yêu kể chuyện cho nghe là gì"
3. "đòi mấy lần toàn kêu không bình thường xịu xịu"

Bước 2: Lập Kế hoạch (Planning)
Tôi sẽ phân tích tuần tự 3 cụm từ trên tương ứng với 3 Opinion riêng biệt để xác định các thành phần cấu trúc của chúng.

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
                        "Intensity": "Standard"
                    },
                    {
                        "Source": [],
                        "Target": [],
                        "Polar_expression": ["không biết người yêu kể chuyện cho nghe là gì"],
                        "Polarity": "Negative",
                        "Intensity": "Standard"
                    },
                    {
                        "Source": [],
                        "Target": [],
                        "Polar_expression": ["đòi mấy lần toàn kêu không bình thường xịu xịu"],
                        "Polarity": "Neutral",
                        "Intensity": "Standard"
                    }
                ]
            }
        },
        {
            "text": "hi vọng câu chuyện admin vừa bịa ra giúp các bạn có thêm niềm tin trong cuộc sống 😂.",
            "reasoning": """Bước 1: Trích xuất Ứng viên (Extraction)
Tôi tìm thấy 1 cụm từ mang sắc thái cảm xúc:
1. "giúp các bạn có thêm niềm tin trong cuộc sống"

Bước 2: Lập Kế hoạch (Planning)
Tôi sẽ phân tích cụm từ trên tương ứng với 1 Opinion để xác định các thành phần cấu trúc của nó.

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
                        "Intensity": "Standard"
                    }
                ]
            }
        },
        # [PLACEHOLDER] Add more examples here
    ]
    
    EXAMPLES_POOL_EN = [
        # [PLACEHOLDER] Add English examples here
    ]

    def __init__(self, eng: bool = False, n_shot: int = 0):
        super().__init__(eng)
        self.n_shot = n_shot
        self.examples_pool = self.EXAMPLES_POOL_EN if eng else self.EXAMPLES_POOL_VI
        self._selected_examples = None

    def prepare(self) -> None:
        """Select examples."""
        if self._is_prepared:
            return
            
        if self.n_shot > 0:
            if self.n_shot > len(self.examples_pool):
                print(f"Warning: Requested {self.n_shot} examples but only {len(self.examples_pool)} available.")
                self._selected_examples = self.examples_pool
            else:
                # Fixed selection for consistency
                self._selected_examples = self.examples_pool[:self.n_shot]
                
        super().prepare()

    def _build_system_prompt(self) -> str:
        """System prompt (Base definitions only)."""
        if self.eng:
            return self._get_system_prompt_en()
        else:
            return self._get_system_prompt_vi()

    def _build_user_prompt(self, text: str, sent_id: str, **kwargs) -> str:
        """Build complex user prompt."""
        if self.eng:
            return self._get_user_prompt_en(text, sent_id)
        else:
            return self._get_user_prompt_vi(text, sent_id)

    # ==================== VIETNAMESE PROMPTS ====================

    def _get_pas_instruction_vi(self) -> str:
        return """Hãy thực hiện một phân tích chi tiết theo quy trình Plan-and-Solve (PaS) sau:

1. [TRÍCH XUẤT ỨNG VIÊN]: Đọc kỹ văn bản. Xác định và liệt kê danh sách các cụm từ (spans) tiềm năng chứa cảm xúc (Polar Expressions) hoặc mô tả hành vi/trạng thái.
2. [LẬP KẾ HOẠCH]: Dựa trên danh sách trên, xác định số lượng Opinion cần phân tích và thứ tự thực hiện.
3. [THỰC THI SUY LUẬN]:
   - Với mỗi Opinion trong kế hoạch, hãy phân tích chi tiết các thành phần: Source, Target, Polar_expression, Polarity, Intensity.
   - BẮT BUỘC: Với mỗi giá trị gán nhãn, phải kèm theo lý giải ngắn gọn (Reasoning) dựa trên ngữ cảnh, tiếng lóng hoặc ẩn ý.
4. [TỔNG HỢP]: Trình bày kết quả cuối cùng dưới dạng JSON."""

    def _build_examples_section_vi(self, examples: List[Dict[str, Any]]) -> str:
        section = "DƯỚI ĐÂY LÀ MỘT SỐ VÍ DỤ MINH HỌA:\n\n"
        for i, example in enumerate(examples, 1):
            section += f"=== VÍ DỤ {i} ===\n"
            section += f'Phân tích cảm xúc cho văn bản sau: "{example["text"]}"\n\n'
            section += f'Đọc lại câu hỏi: \nPhân tích cảm xúc cho văn bản sau: "{example["text"]}"\n\n'
            section += f'Reasoning:\n{example["reasoning"]}\n\n'
            section += f'Output:\n{json.dumps(example["output"], ensure_ascii=False, indent=2)}\n\n'
        return section.strip()

    def _get_user_prompt_vi(self, text: str, sent_id: str) -> str:
        # 1. PaS Instruction
        parts = [self._get_pas_instruction_vi()]
        
        # 2. CoT Examples
        if self.n_shot > 0 and self._selected_examples:
            parts.append(self._build_examples_section_vi(self._selected_examples))
            
        # 3. Base Question
        base_question = f"""Phân tích cảm xúc cho văn bản sau (sent_id: {sent_id}):
"{text}"
"""
        parts.append(base_question)
        
        # 4. Re-reading instruction
        parts.append(f"""Đọc lại câu hỏi: {base_question}""")
                
        return "\n\n".join(parts)

    def _get_system_prompt_vi(self) -> str:
        return """Bạn là chuyên gia trong lĩnh vực phân tích cảm xúc tiếng Việt có cấu trúc. 
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

"""

    # ==================== ENGLISH PROMPTS (Placeholders) ====================
    
    def _get_user_prompt_en(self, text: str, sent_id: str) -> str:
        return f"""Analyze sentiment for (sent_id: {sent_id}): "{text}" """

    def _get_system_prompt_en(self) -> str:
        return "You are a sentiment analysis expert."