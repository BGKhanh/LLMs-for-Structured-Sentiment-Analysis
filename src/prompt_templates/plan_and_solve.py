# src/prompt_templates/plan_and_solve.py

from typing import Tuple
from .base import BasePromptTemplate


class PlanAndSolvePrompt(BasePromptTemplate):
    """
    Plan-and-Solve prompt technique for structured sentiment analysis.
    
    Paper: Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning
    Link: https://arxiv.org/abs/2305.04091
    
    Design:
    - Two modes: PS (basic) and PS+ (enhanced)
    - System prompt same as zero-shot
    - User prompt varies based on mode (plus parameter)
    - No examples, single stage
    - PS: "Let's first understand the problem and devise a plan to solve it. Then, let's carry out the plan to solve the problem."
    - PS+: Adds more detailed instructions (extract relevant info, calculate carefully, etc.)
    """
    
    def __init__(self, eng: bool = False, plus: bool = False):
        """
        Initialize Plan-and-Solve prompt generator.
        
        Args:
            eng: If True, use English prompts. If False, use Vietnamese prompts.
            plus: If True, use PS+ (enhanced). If False, use PS (basic).
        
        Note:
            PS+ adds more detailed reasoning instructions compared to PS.
        """
        super().__init__(eng)
        self.plus = plus
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt (same as zero-shot, shared for PS and PS+).
        
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
        Build user prompt based on mode (PS or PS+).
        
        Args:
            text: Text to analyze
            sent_id: Sample identifier
        
        Returns:
            User prompt with PS or PS+ instructions
        """
        if self.eng:
            if self.plus:
                return self._get_user_prompt_ps_plus_en(text, sent_id)
            else:
                return self._get_user_prompt_ps_en(text, sent_id)
        else:
            if self.plus:
                return self._get_user_prompt_ps_plus_vi(text, sent_id)
            else:
                return self._get_user_prompt_ps_vi(text, sent_id)
    
    # ==================== SYSTEM PROMPTS (SHARED) ====================
    
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
    
    # ==================== VIETNAMESE USER PROMPTS ====================
    
    def _get_user_prompt_ps_vi(self, text: str, sent_id: str) -> str:
        """Vietnamese user prompt for PS (basic)."""
        return f"""Phân tích cảm xúc cho văn bản sau (sent_id: {sent_id}):
"{text}"

Hãy cùng phân tích bài toán này. Trước tiên hãy hiểu vấn đề và lập kế hoạch giải quyết. Sau đó, thực hiện kế hoạch để giải quyết vấn đề theo các bước sau:
Đầu tiên, hãy vạch ra một kế hoạch để xác định tất cả các opinions và các thành phần của chúng trong câu.
Sau đó, thực hiện kế hoạch đó từng bước một để đưa ra phân tích chi tiết.
Cuối cùng, tổng hợp tất cả các phân tích vào một khối JSON duy nhất.
"""
    
    def _get_user_prompt_ps_plus_vi(self, text: str, sent_id: str) -> str:
        """Vietnamese user prompt for PS+ (enhanced)."""
        return f"""Phân tích cảm xúc cho văn bản sau (sent_id: {sent_id}):
"{text}"

Hãy thực hiện một phân tích chi tiết theo các bước sau:
1.  **Phân tích Sơ bộ & Trích xuất Dấu hiệu:** Đọc kỹ văn bản. Trích xuất tất cả các cụm từ khóa, đại từ nhân xưng, tiếng lóng, và các dấu hiệu ngôn ngữ có thể liên quan đến cảm xúc.
2.  **Lập Kế hoạch Phân tích:** Dựa trên các dấu hiệu đã trích xuất, vạch ra một kế hoạch để phân tích tuần tự từng Opinion.
3.  **Thực thi Kế hoạch:**
    -   Với mỗi Opinion, hãy xác định giá trị cho từng thành phần (Source, Target, Polar_expression, Polarity, Intensity).
    -   Hãy giải thích ngắn gọn lý do của bạn, đặc biệt chú ý đến ngữ cảnh, ẩn ý và các quy tắc đã được định nghĩa.
4.  **Tổng hợp Kết quả:** Xây dựng khối JSON cuối cùng dựa trên toàn bộ phân tích ở trên.
"""
    
    # ==================== ENGLISH USER PROMPTS ====================
    
    def _get_user_prompt_ps_en(self, text: str, sent_id: str) -> str:
    """English user prompt for PS (basic)."""
    return f"""Analyze the sentiment for the following text (sent_id: {sent_id}):
"{text}"

Let's break down this problem. First, let's understand the issue and plan the solution. Then, execute the plan to solve the problem in the following steps:
First, outline a plan to identify all opinions and their components in the sentence.
Next, execute that plan step-by-step to provide a detailed analysis.
Finally, summarize all analyses into a single JSON block.
"""

def _get_user_prompt_ps_plus_en(self, text: str, sent_id: str) -> str:
    """English user prompt for PS+ (enhanced)."""
    return f"""Analyze the sentiment for the following text (sent_id: {sent_id}):
"{text}"

Perform a detailed analysis following these steps:
1. **Preliminary Analysis & Signal Extraction:** Read the text carefully. Extract all keywords, personal pronouns, slang (teencode), and linguistic signals that might relate to sentiment.
2. **Analysis Planning:** Based on the extracted signals, outline a plan to sequentially analyze each Opinion.
3. **Plan Execution:**
    - For each Opinion, determine the value for each component (Source, Target, Polar_expression, Polarity, Intensity).
    - Briefly explain your reasoning, paying special attention to context, implicit meaning, and the defined rules.
4. **Result Synthesis:** Construct the final JSON block based on the entire analysis above.
"""


# # Example usage:
# # PS (basic)
# ps = PlanAndSolvePrompt(eng=False, plus=False)
# ps.prepare()
# 
# # PS+ (enhanced)
# ps_plus = PlanAndSolvePrompt(eng=False, plus=True)
# ps_plus.prepare()
# 
# # Use with SentimentDataset
# test_dataset = SentimentDataset(
#     data_path="data/test.json",
#     prompt_generator=ps_plus.get_prompt
# )