# src/prompt_templates/re_reading.py

from typing import Tuple
from .base import BasePromptTemplate


class ReReadingPrompt(BasePromptTemplate):
    """
    Re-reading prompt technique for structured sentiment analysis.
    
    Paper: RE-READING Improves Reasoning in Large Language Models
    Link: https://arxiv.org/abs/2309.06275
    
    Design:
    - Simple technique: Ask the question twice
    - System prompt same as zero-shot
    - User prompt repeats the question
    - No examples, no CoT reasoning
    - Single stage
    """
    
    def __init__(self, eng: bool = False, add_method: Literal["none", "CoT", "PaS"] = "none"):
        """
        Initialize Re-reading prompt generator.
        
        Args:
            eng: If True, use English prompts. If False, use Vietnamese prompts.
            add_method: Enhancement method to combine with Re-reading.
                        - "none": Vanilla Re-reading (RE2)
                        - "CoT": Chain-of-Thought (RE2+CoT)
                        - "PaS": Plan-and-Solve (RE2+PaS)
        
        Note:
            This is the simplest prompting technique - just repeats the question.
        """
        super().__init__(eng)
        self.add_method = add_method
    def _build_system_prompt(self) -> str:
        """
        Build system prompt (same as zero-shot).
        
        Returns:
            System prompt with component definitions
        """
        if self.eng:
            return self._get_system_prompt_en()
        else:
            return self._get_system_prompt_vi()
    
    def _build_user_prompt(
        self,
        text: str,
        sent_id: str,
        **kwargs
    ) -> str:
        """
        Build user prompt with re-reading (question repeated twice).
        
        Args:
            text: Text to analyze
            sent_id: Sample identifier
        
        Returns:
            User prompt with question repeated twice
        """
        if self.eng:
            return self._get_user_prompt_en(text, sent_id)
        else:
            return self._get_user_prompt_vi(text, sent_id)
    
    
    def _get_trigger_vi(self) -> str:
        """Get the Vietnamese trigger phrase based on add_method."""
        if self.add_method == "CoT":
            return "A: Hãy cùng suy nghĩ từng bước."
        elif self.add_method == "PaS":
            # Điều chỉnh [Solving/Calculations] thành [Phân tích chi tiết] cho hợp ngữ cảnh
            return ("A: Đầu tiên hãy hiểu vấn đề và vạch ra kế hoạch để giải quyết. "
                    "Sau đó, hãy thực hiện kế hoạch, phân tích từng bước, "
                    "và đưa ra câu trả lời cuối cùng. "
                    "Vui lòng sinh ra quy trình cụ thể theo các bước: "
                    "[Hiểu vấn đề], [Lập kế hoạch], [Phân tích chi tiết], [Câu trả lời].")
        return ""

    def _get_trigger_en(self) -> str:
        """Get the English trigger phrase based on add_method."""
        if self.add_method == "CoT":
            return "A: Let's think step by step."
        elif self.add_method == "PaS":
            # Adapted from Table 11  but tuned for extraction
            return ("A: Let's first understand the problem and devise a plan to solve the problem. "
                    "Then, let's carry out the plan, solve the problem step by step, "
                    "and give the ultimate answer. Please explicitly generate the mentioned process: "
                    "[Problem Understanding], [Plan], [Detailed Analysis], [Answer].")
        return ""
    
    # ==================== VIETNAMESE PROMPTS ====================
    
    def _get_system_prompt_vi(self) -> str:
        """Vietnamese system prompt (same as zero-shot)."""
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
    
    def _get_user_prompt_vi(self, text: str, sent_id: str) -> str:
        """Vietnamese user prompt with RE2 + logic."""
        
        # 1. Base Question (Pass 1)
        base_question = f"""Phân tích cảm xúc cho văn bản sau (sent_id: {sent_id}):
"{text}"

Trả về KẾT QUẢ CHÍNH XÁC theo cấu trúc JSON đã yêu cầu."""
        
        # 2. Re-reading instruction (Pass 2) 
        re_reading_part = f"Đọc lại câu hỏi: {base_question}"
        
        # 3. Add Method Trigger (CoT or PaS)
        trigger = self._get_trigger_vi()
        
        # Combine parts
        full_prompt = f"{base_question}\n\n{re_reading_part}"
        
        if trigger:
            full_prompt += f"\n\n{trigger}"
            
        return full_prompt
    
    # ==================== ENGLISH PROMPTS ====================
    
    def _get_system_prompt_en(self) -> str:
        """English system prompt (same as zero-shot)."""
        return """You are an expert in structured Vietnamese sentiment analysis. 
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

"""
    
    def _get_user_prompt_en(self, text: str, sent_id: str) -> str:
        """English user prompt with RE2 + logic."""
        
        # 1. Base Question (Pass 1)
        base_question = f"""Analyze the sentiment for the following text (sent_id: {sent_id}):
"{text}"

Return the EXACT RESULT according to the requested JSON structure."""
        
        # 2. Re-reading instruction (Pass 2) 
        re_reading_part = f"Read the question again: {base_question}"
        
        # 3. Add Method Trigger (CoT or PaS)
        trigger = self._get_trigger_en()
        
        # Combine parts
        full_prompt = f"{base_question}\n\n{re_reading_part}"
        
        if trigger:
            full_prompt += f"\n\n{trigger}"
            
        return full_prompt


# # Example usage:
# re_reading = ReReadingPrompt(eng=False)
# re_reading.prepare()  # Cache system prompt
# 
# # Use with SentimentDataset
# test_dataset = SentimentDataset(
#     data_path="data/test.json",
#     prompt_generator=re_reading.get_prompt
# )