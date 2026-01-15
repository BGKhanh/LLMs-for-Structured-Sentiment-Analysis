# src/prompt_templates/plan_and_solve.py

import json
import random
from typing import Tuple, List, Dict, Any
from .base import BasePromptTemplate
from .re2_pas_cot import Re2PaSCoTPrompt


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
    
    def __init__(self, eng: bool = False, plus: bool = False, n_shot: int = 0):

        """
        Initialize Plan-and-Solve prompt generator.
        
        Args:
            eng: If True, use English prompts. If False, use Vietnamese prompts.
            plus: If True, use PS+ (enhanced). If False, use PS (basic).
            n_shot: Number of examples (only used if plus=True).
        
        Note:
            PS+ adds more detailed reasoning instructions compared to PS.
        """
        super().__init__(eng)
        super().__init__(eng)
        self.plus = plus
        self.n_shot = n_shot
        self._selected_examples = None

    def prepare(self) -> None:
        """Select examples if in PS+CoT mode."""
        if self._is_prepared:
            return
            
        # Only load examples if plus=True and n_shot > 0 (PaS+CoT mode)
        if self.plus and self.n_shot > 0:
            # Use pool from Re2PaSCoTPrompt as requested
            pool = Re2PaSCoTPrompt.EXAMPLES_POOL_EN if self.eng else Re2PaSCoTPrompt.EXAMPLES_POOL_VI
            
            if len(pool) < self.n_shot:
                print(f"⚠️ Warning: Requested {self.n_shot} shots but pool only has {len(pool)}. Using all.")
                self._selected_examples = pool
            else:
                self._selected_examples = pool[:self.n_shot]
                
            print(f"✅ Selected {len(self._selected_examples)} PaS-style examples for Plan-and-Solve")
            
        super().prepare()
                    
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

    def _format_pas_example(self, example: Dict[str, Any]) -> str:
        """Format a single example with PaS reasoning."""
        return f'Input: "{example.get("text", "")}"\nReasoning:\n{example.get("reasoning", "")}\nOutput:\n{json.dumps(example.get("output", {}), ensure_ascii=False, indent=2)}'

    def _build_examples_section(self, examples: List[Dict[str, Any]], title: str) -> str:
        """Build examples section."""
        section = f"{title}\n\n"
        for i, example in enumerate(examples, 1):
            section += f"=== EXAMPLE {i} ===\n"
            section += self._format_pas_example(example)
            section += "\n\n"
        return section.strip()
        
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
        return f"""
Đầu tiên hãy hiểu vấn đề và vạch ra kế hoạch để giải quyết. 
Sau đó, hãy thực hiện kế hoạch, phân tích từng bước và đưa ra câu trả lời cuối cùng. 
Vui lòng sinh ra quy trình cụ thể theo các bước: 
[Hiểu vấn đề], [Lập kế hoạch], [Phân tích chi tiết], [Câu trả lời].

Phân tích cảm xúc cho văn bản sau (sent_id: {sent_id}):
"{text}"
"""
    
    def _get_user_prompt_ps_plus_vi(self, text: str, sent_id: str) -> str:
        """Vietnamese user prompt for PS+ (enhanced) & PaS+CoT."""
        
        # 1. Instruction
        instruction = """Hãy thực hiện một phân tích chi tiết theo quy trình Plan-and-Solve (PaS) sau:

1. [TRÍCH XUẤT ỨNG VIÊN]: Đọc kỹ văn bản. Xác định và liệt kê danh sách các cụm từ (spans) tiềm năng chứa cảm xúc (Polar Expressions) hoặc mô tả hành vi/trạng thái.
2. [LẬP KẾ HOẠCH]: 
   - Rà soát lại danh sách ứng viên ở Bước 1. Loại bỏ các cụm từ không rõ ràng hoặc trùng lặp (nếu có).
   - Nêu chiến lược xử lý: thứ tự phân tích, cách tiếp cận các thành phần. Xác định Source/Target nếu chúng bị ẩn.
   - Nhận diện các đặc điểm chung/thách thức có thể gặp phải trong văn bản này.
   - Xác nhận sẽ phân tích đầy đủ 5 thành phần cho mỗi Opinion.
3. [THỰC THI SUY LUẬN]:
   - Với mỗi Opinion trong kế hoạch, hãy phân tích chi tiết các thành phần: Source, Target, Polar_expression, Polarity, Intensity.
   - BẮT BUỘC: Với mỗi giá trị gán nhãn, phải kèm theo lý giải ngắn gọn (Reasoning) dựa trên ngữ cảnh, tiếng lóng hoặc ẩn ý.
4. [TỔNG HỢP]: Trình bày kết quả cuối cùng dưới dạng JSON.
"""
        parts = [instruction]

        # 2. Examples (only for PaS+CoT)
        if self.n_shot > 0 and self._selected_examples:
            examples_text = self._build_examples_section(
                self._selected_examples, 
                "DƯỚI ĐÂY LÀ MỘT SỐ VÍ DỤ MINH HỌA (HÃY LÀM THEO QUY TRÌNH TƯƠNG TỰ):"
            )
            parts.append(examples_text)
            parts.append("\nBây giờ, hãy phân tích trường hợp sau:\n")

        # 3. Input Question
        question = f"""Phân tích cảm xúc cho văn bản sau (sent_id: {sent_id}):
"{text}"
"""
        parts.append(question)
        
        return "\n".join(parts)
    
    # ==================== ENGLISH USER PROMPTS ====================
    
    def _get_user_prompt_ps_en(self, text: str, sent_id: str) -> str:
        """English user prompt for PS (basic)."""
        return f"""
Let's first understand the problem and devise a plan to solve the problem. 
Then, let's carry out the plan, solve the problem step by step and give the ultimate answer.
Please explicitly generate the mentioned process: 
[Problem Understanding], [Plan], [Detailed Analysis], [Answer].

Analyze the sentiment for the following text (sent_id: {sent_id}):
"{text}"
"""

    def _get_user_prompt_ps_plus_en(self, text: str, sent_id: str) -> str:
        """English user prompt for PS+ (enhanced) & PaS+CoT."""
        
        # 1. Instruction
        instruction = """Perform a detailed analysis following these steps:
1. **Preliminary Analysis & Signal Extraction:** Read the text carefully. Extract all keywords, personal pronouns, slang (teencode), and linguistic signals that might relate to sentiment.
2. **Analysis Planning:** Based on the extracted signals, outline a plan to sequentially analyze each Opinion.
3. **Plan Execution:**
    - For each Opinion, determine the value for each component (Source, Target, Polar_expression, Polarity, Intensity).
    - Briefly explain your reasoning, paying special attention to context, implicit meaning, and the defined rules.
4. **Result Synthesis:** Construct the final JSON block based on the entire analysis above.
"""
        parts = [instruction]

        # 2. Examples (only for PaS+CoT)
        if self.n_shot > 0 and self._selected_examples:
            examples_text = self._build_examples_section(
                self._selected_examples, 
                "HERE ARE SOME DEMONSTRATION EXAMPLES (PLEASE FOLLOW SIMILAR PROCESS):"
            )
            parts.append(examples_text)
            parts.append("\nNow, analyze the following case:\n")

        # 3. Input Question
        question = f"""Analyze the sentiment for the following text (sent_id: {sent_id}):
"{text}"
"""
        parts.append(question)
        
        return "\n".join(parts)


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