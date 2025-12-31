# src/prompt_templates/few_shot_CoT.py

from typing import Dict, Any, List, Tuple
from .base import BasePromptTemplate
from .re2_pas_cot import Re2PaSCoTPrompt
import json

class FewShotCoTPrompt(BasePromptTemplate):
    """
    Few-shot Chain-of-Thought prompt generator.
    
    Combines few-shot learning with chain-of-thought reasoning.
    Uses hardcoded examples with pre-written reasoning demonstrations.
    """
    
    # ... (Giữ nguyên phần EXAMPLES_POOL_VI và EXAMPLES_POOL_EN - không thay đổi dữ liệu) ...
    # Hardcoded examples pool with reasoning
    EXAMPLES_POOL_VI = Re2PaSCoTPrompt.EXAMPLES_POOL_VI
    EXAMPLES_POOL_EN = Re2PaSCoTPrompt.EXAMPLES_POOL_EN
    
    def __init__(self, eng: bool = False, n_shot: int = 3):
        """
        Initialize Few-shot CoT prompt generator.
        
        Args:
            eng: If True, use English prompts and examples. 
                 If False, use Vietnamese.
            n_shot: Number of examples to include (default: 3).
                    Must be <= number of examples in pool.
        """
        super().__init__(eng)
        
        if n_shot < 0:
            raise ValueError("n_shot must be >= 0")
        
        self.n_shot = n_shot
        self._selected_examples = None  # Selected examples (fixed for all samples)
    
    def prepare(self) -> None:
        """Select examples and build system prompt with demonstrations."""
        if self._is_prepared:
            return
        
        # Select examples from hardcoded pool
        if self.n_shot > 0:
            self._select_examples()
        
        # Parent class will build and cache system prompt
        super().prepare()
    
    def _select_examples(self) -> None:
        """Select n_shot examples from hardcoded pool."""
        examples_pool = self.EXAMPLES_POOL_EN if self.eng else self.EXAMPLES_POOL_VI
        
        if self.n_shot > len(examples_pool):
            raise ValueError(
                f"Requested n_shot={self.n_shot} but only "
                f"{len(examples_pool)} examples available in pool"
            )
        
        # Take first n examples (Fixed selection)
        self._selected_examples = examples_pool[:self.n_shot]
        print(f"✅ Selected {self.n_shot} hardcoded examples for few-shot CoT")
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt (Clean definition only).
        Matches behavior of few_shot.py: No examples here.
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
        Build user prompt for current sample.
        Includes examples here (User Prompt) instead of System Prompt.
        """
        if self.eng:
            return self._get_user_prompt_en(text, sent_id)
        else:
            return self._get_user_prompt_vi(text, sent_id)
    
    # ==================== VIETNAMESE PROMPTS ====================
    
    def _get_system_prompt_vi(self) -> str:
        """Vietnamese system prompt with task definition only."""
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
        """Vietnamese user prompt with examples (if any)."""
        
        prompt_parts = []
        
        # 1. Examples (if any)
        if self.n_shot > 0 and self._selected_examples:
            examples_text = self._build_examples_section_vi(self._selected_examples)
            prompt_parts.append(examples_text)
            prompt_parts.append("\nBây giờ, hãy phân tích trường hợp sau:\n")
            
        # 2. Input
        input_section = f"""Input: "{text}" (sent_id: {sent_id})"""
        
        prompt_parts.append(input_section)
        
        return "".join(prompt_parts)
    
    def _build_examples_section_vi(
        self,
        examples: List[Dict[str, Any]]
    ) -> str:
        """Build Vietnamese examples section with reasoning."""
        section = "DƯỚI ĐÂY LÀ MỘT SỐ VÍ DỤ MINH HỌA:\n\n"
        
        for i, example in enumerate(examples, 1):
            section += f"=== VÍ DỤ {i} ===\n"
            section += f'Input: "{example["text"]}"\n\n'
            section += f'Reasoning:\n{example["reasoning"]}\n\n'
            section += f'Output:\n'
            
            # Format output as JSON
            section += json.dumps(example["output"], ensure_ascii=False, indent=2)
            section += "\n\n"
        
        return section.strip()
    
    # ==================== ENGLISH PROMPTS ====================
    
    def _get_system_prompt_en(self) -> str:
        """English system prompt with task definition only."""
        return """You are a Vietnamese sentiment analysis expert.
Your task is to analyze the comments, present your reasoning process, and finally provide the result in a single JSON block. 
Please follow the examples provided (if any).
"""
    
    def _get_user_prompt_en(self, text: str, sent_id: str) -> str:
        """English user prompt with examples (if any)."""
        prompt_parts = []
        
        # 1. Examples
        if self.n_shot > 0 and self._selected_examples:
            examples_text = self._build_examples_section_en(self._selected_examples)
            prompt_parts.append(examples_text)
            prompt_parts.append("\nNow, analyze the following case:\n")
            
        # 2. Input
        input_section = f"""Input: "{text}" (sent_id: {sent_id})

Think step by step, then return the final JSON."""
        prompt_parts.append(input_section)
        
        return "".join(prompt_parts)
    
    def _build_examples_section_en(
        self,
        examples: List[Dict[str, Any]]
    ) -> str:
        """Build English examples section with reasoning."""
        section = "HERE ARE SOME DEMONSTRATION EXAMPLES:\n\n"
        
        for i, example in enumerate(examples, 1):
            section += f"=== EXAMPLE {i} ===\n"
            section += f'Input: "{example["text"]}"\n\n'
            section += f'Reasoning:\n{example["reasoning"]}\n\n'
            section += f'Output:\n'
            
            # Format output as JSON
            section += json.dumps(example["output"], ensure_ascii=False, indent=2)
            section += "\n\n"
        
        return section.strip()