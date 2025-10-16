from typing import Dict, Any, List, Tuple
from .base import BasePromptTemplate


class ZeroShotCoTPrompt(BasePromptTemplate):
    """
    Zero-shot Chain-of-Thought prompt generator for structured sentiment analysis.
    Paper: Large Language Models are Zero-Shot Reasoners (Zero-shot CoT)
    Link: https://arxiv.org/pdf/2205.11916
    
    Implements two-stage CoT approach:
    - Stage 1: Generate reasoning process
    - Stage 2: Extract structured JSON output based on reasoning
    """
    
    def __init__(self, eng: bool = False):
        """
        Initialize Zero-shot CoT prompt generator.
        
        Args:
            eng: If True, use English prompts. If False, use Vietnamese prompts.
        """
        super().__init__(eng)
        
        # Cache for both stages (instead of single _system_prompt_cache)
        self._stage1_system_prompt = None
        self._stage2_system_prompt = None
        
    def prepare(self) -> None:
        """Build and cache both stage prompts."""
        if self._is_prepared:
            return
        
        # Cache both stages
        if self.eng:
            self._stage1_system_prompt = self._get_stage1_system_prompt_en()
            self._stage2_system_prompt = self._get_stage2_system_prompt_en()
        else:
            self._stage1_system_prompt = self._get_stage1_system_prompt_vi()
            self._stage2_system_prompt = self._get_stage2_system_prompt_vi()
        
        self._is_prepared = True
        print(f"✅ {self.__class__.__name__} prepared (both stages cached)")

    def _build_system_prompt(self) -> str:
        """
        Not used for ZeroShotCoT (we use stage-specific prompts).
        
        Required by base class but overridden by get_prompt().
        """
        return self._stage1_system_prompt if self._stage1_system_prompt else ""
    
    def _build_user_prompt(self, text: str, sent_id: str, **kwargs) -> str:
        """
        Not used for ZeroShotCoT (we override get_prompt()).
        
        Required by base class but overridden by get_prompt().
        """
        return ""

    def get_prompt(
        self,
        text: str,
        sent_id: str,
        stage: str = "stage_1",
        reasoning: str = None
    ) -> Tuple[str, str]:
        """
        Get prompt based on stage.
        
        Args:
            text: Text to analyze
            sent_id: Sentence identifier
            stage: "stage_1" (reasoning) or "stage_2" (JSON extraction)
            reasoning: Reasoning output from stage 1 (required for stage_2)
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        
        Raises:
            ValueError: If stage is invalid or reasoning is missing for stage_2
        """
        # Lazy preparation
        if not self._is_prepared:
            self.prepare()
        
        # Handle None stage
        if stage is None:
            stage = "stage_1"
        
        # Validate stage
        valid_stages = ["stage_1", "stage_2"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")
        
        # Route to appropriate stage
        if stage == "stage_1":
            system_prompt = self._stage1_system_prompt
            user_prompt = self._get_stage1_user_prompt(text, sent_id)
        else:  # stage_2
            if reasoning is None:
                raise ValueError("Reasoning is required for stage_2")
            system_prompt = self._stage2_system_prompt
            user_prompt = self._get_stage2_user_prompt(text, sent_id, reasoning)
        
        return system_prompt, user_prompt
    
    # ==================== STAGE 1 PROMPTS ====================


    def _get_stage1_system_prompt_vi(self) -> str:
        """Vietnamese system prompt for Stage 1 (reasoning)."""
        return """Bạn là chuyên gia trong lĩnh vực phân tích cảm xúc tiếng Việt có cấu trúc. 
Nhiệm vụ của bạn là phân tích bình luận mạng xã hội và trích xuất các thành phần cảm xúc. 
Giải thích chi tiết quá trình suy luận của bạn để tìm ra các thành phần cảm xúc.

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

YÊU CẦU:
- Với mỗi bình luận, hãy phân tích tuần tự từng opinion mà bạn tìm thấy. Bắt đầu mỗi opinion bằng "--- Phân tích Opinion 1 ---", "--- Phân tích Opinion 2 ---", v.v.
- Với mỗi opinion, giải thích rõ ràng cách bạn tìm thấy từng thành phần: Source, Target, Polar_expression, Polarity, và Intensity."""
    
    def _get_stage1_user_prompt(self, text: str, sent_id: str) -> str:
        """User prompt for Stage 1 (reasoning)."""
        if self.eng:
            return f"""Let's think step by step. Analyze the sentiment for the following text (sent_id: {sent_id}):
"{text}"
"""
        else:
            return f"""Hãy suy luận theo từng bước một. Phân tích cảm xúc cho văn bản sau (sent_id: {sent_id}):
"{text}"
"""
    
    def _get_stage1_system_prompt_en(self) -> str:
        """English system prompt for Stage 1 (reasoning)."""
        return """You are an expert in structured Vietnamese sentiment analysis. 
Your task is to analyze social media comments and extract sentiment components. 
Provide a detailed explanation of your reasoning process to find the sentiment components.

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

REQUIREMENTS:
- For each comment, analyze each opinion you find sequentially. Start each opinion with "--- Opinion 1 Analysis ---", "--- Opinion 2 Analysis ---", etc.
- For each opinion, clearly explain how you found each component: Source, Target, Polar_expression, Polarity, and Intensity.
"""
  
    # ==================== STAGE 2 PROMPTS ====================

   def _get_stage2_system_prompt_vi(self) -> str:
        """Vietnamese system prompt for Stage 2 (JSON extraction)."""
        return """Bạn là một công cụ xử lý văn bản có nhiệm vụ chuyển đổi một bài phân tích thành cấu trúc JSON được chỉ định.

JSON OUTPUT FORMAT:
{
  "sent_id": "[ID của câu]",
  "text": "[Bình luận gốc]",
  "opinions": [
    {
      "Source": ["text_span_1", "text_span_2"],
      "Target": ["text_span_1", "text_span_2"],
      "Polar_expression": ["text_span_1", "text_span_2"],
      "Polarity": "Positive/Negative/Neutral",
      "Intensity": "Strong/Standard/Weak"
    }
  ]
}

QUAN TRỌNG: 
- CHỈ TRẢ VỀ MỘT KHỐI JSON HOÀN CHỈNH.
- KHÔNG THÊM BẤT KỲ GIẢI THÍCH, BÌNH LUẬN, HAY VĂN BẢN NÀO KHÁC BÊN NGOÀI KHỐI JSON.
- Đảm bảo các giá trị text_span được trích xuất chính xác từ văn bản gốc.
"""


   def _get_stage2_user_prompt(
        self,
        text: str,
        sent_id: str,
        reasoning: str
    ) -> str:
        """User prompt for Stage 2 (JSON extraction)."""
        if self.eng:
            return f"""Based on the information below, generate the final JSON.

--- ORIGINAL TEXT ---
sent_id: {sent_id}
text: "{text}"

--- REASONING ANALYSIS ---
{reasoning}

--- FINAL JSON ---
"""
        else:
            return f"""Dựa vào các thông tin dưới đây, hãy tạo ra JSON cuối cùng.

--- VĂN BẢN GỐC ---
sent_id: {sent_id}
text: "{text}"

--- PHÂN TÍCH SUY LUẬN ---
{reasoning}

--- JSON CUỐI CÙNG ---
""" 
    # ==================== STAGE 2 PROMPTS ====================
    
    def _get_stage2_system_prompt_en(self) -> str:
        """English system prompt for Stage 2 (JSON extraction)."""
        return """You are a text processing tool tasked with converting an analysis into a specified JSON structure.

JSON OUTPUT FORMAT:
{
  "sent_id": "[Sentence ID]",
  "text": "[Original comment]",
  "opinions": [
    {
      "Source": ["text_span_1", "text_span_2"],
      "Target": ["text_span_1", "text_span_2"],
      "Polar_expression": ["text_span_1", "text_span_2"],
      "Polarity": "Positive/Negative/Neutral",
      "Intensity": "Strong/Standard/Weak"
    }
  ]
}

IMPORTANT: 
- RETURN ONLY ONE COMPLETE JSON BLOCK.
- DO NOT ADD ANY EXPLANATIONS, COMMENTS, OR OTHER TEXT OUTSIDE THE JSON BLOCK.
- Ensure that the text_span values are extracted exactly from the original text.
"""



# # Example usage:
# # Stage 1
# cot = ZeroShotCoTPrompt(eng=False)
# cot.prepare()  # Cache both stage prompts
# 
# sys_prompt, user_prompt = cot.get_prompt(
#     text="Example text",
#     sent_id="001",
#     stage="stage_1"
# )
# 
# # ... run model to get reasoning ...
# reasoning_output = "..."
# 
# # Stage 2
# sys_prompt, user_prompt = cot.get_prompt(
#     text="Example text",
#     sent_id="001",
#     stage="stage_2",
#     reasoning=reasoning_output
# )