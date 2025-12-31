# src/prompt_templates/re_reading.py

import json
import random
from typing import Tuple, Literal, Optional, List, Dict, Any
from torch.utils.data import Dataset
from .base import BasePromptTemplate
from .re2_pas_cot import Re2PaSCoTPrompt

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
    
    def __init__(
        self, 
        eng: bool = False, 
        add_method: Literal["none", "0_CoT", "FewShot", "FewShot_CoT", "PaS"] = "none",
        n_shot: int = 0,
        examples_pool: Optional[Dataset] = None,
        examples_pool_path: Optional[str] = None
    ):
        """
        Initialize Re-reading prompt generator.
        
        Args:
            eng: If True, use English prompts. If False, use Vietnamese prompts.
            add_method: Enhancement method to combine with Re-reading.
                        - "none": Vanilla Re-reading (RE2)
                        - "0_CoT": Chain-of-Thought (RE2+0_CoT)
                        - "FewShot": Few-shot (RE2+FewShot)
                        - "FewShot_CoT": Few-shot + Chain-of-Thought (RE2+FewShot_CoT)
                        - "PaS": Plan-and-Solve (RE2+PaS)
            n_shot: Number of examples for FewShot methods.
            examples_pool: Dataset object containing example pool.
            examples_pool_path: Path to JSON file containing example pool.
        Note:
            This is the simplest prompting technique - just repeats the question.
        """
        super().__init__(eng)
        self.add_method = add_method
        
        # Few-shot params
        self.n_shot = n_shot
        self.examples_pool_path = examples_pool_path
        self._examples_pool_raw = examples_pool
        
        self._examples_pool = None
        self._selected_examples = None
    
    def prepare(self) -> None:
        """Load examples pool and select fixed examples if needed."""
        if self._is_prepared:
            return
        
        # Load examples if FewShot method is used
        if self.n_shot > 0:
            if self.add_method == "FewShot":
                self._load_examples_pool()
                self._select_fixed_examples()
            elif self.add_method == "FewShot_CoT":
                # Use hardcoded pool from Re2PaSCoTPrompt
                pool = Re2PaSCoTPrompt.EXAMPLES_POOL_EN if self.eng else Re2PaSCoTPrompt.EXAMPLES_POOL_VI
                if len(pool) < self.n_shot:
                    print(f"⚠️ Warning: Requested {self.n_shot} shots but CoT pool only has {len(pool)}. Using all.")
                    self._selected_examples = pool
                else:
                    # Lấy n_shot đầu tiên (Fixed)
                    self._selected_examples = pool[:self.n_shot]
                print(f"✅ Selected {len(self._selected_examples)} hardcoded examples for Re-reading FewShot_CoT")
            
        super().prepare()
     
    def _select_fixed_examples(self) -> None:
        """Select fixed examples for all samples (Standard FewShot)."""
        if not self._examples_pool:
            raise ValueError("Examples pool is empty.")
            
        if len(self._examples_pool) < self.n_shot:
            raise ValueError(
                f"Not enough examples. Requested {self.n_shot}, "
                f"but only {len(self._examples_pool)} available."
            )
        
        # Random sample cố định cho toàn bộ quá trình inference
        self._selected_examples = random.sample(self._examples_pool, self.n_shot)
        print(f"✅ Selected {self.n_shot} fixed examples for Re-reading FewShot")

    def _load_examples_pool(self) -> None:
        """Load examples from provided source."""
        # Priority 1: Raw list
        if self._examples_pool_raw is not None:
            self._examples_pool = self._examples_pool_raw
            return
        
        # Priority 2: File path
        if self.examples_pool_path is not None:
            try:
                with open(self.examples_pool_path, 'r', encoding='utf-8') as f:
                    self._examples_pool = json.load(f)
                return
            except Exception as e:
                raise ValueError(f"Error loading examples pool: {e}")
        
        # No source provided
        raise ValueError("n_shot > 0 but no examples pool provided (examples_pool or examples_pool_path).")
            
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
        if self.add_method == "0_CoT":
            return "Hãy cùng suy nghĩ từng bước."
        elif self.add_method == "PaS":
            return ("Đầu tiên hãy hiểu vấn đề và vạch ra kế hoạch để giải quyết. "
                    "Sau đó, hãy thực hiện kế hoạch, phân tích từng bước, "
                    "và đưa ra câu trả lời cuối cùng. "
                    "Vui lòng sinh ra quy trình cụ thể theo các bước: "
                    "[Hiểu vấn đề], [Lập kế hoạch], [Phân tích chi tiết], [Câu trả lời].")
        return ""

    def _format_example_re2_vi(self, example: Dict[str, Any], include_reasoning: bool = False) -> str:
        """Format a single example with Re-reading logic (Vietnamese)."""
        text = example.get('text', '')
        
        # Base Question
        q1 = f'Phân tích cảm xúc cho văn bản sau: "{text}"'
        
        # Re-reading
        q2 = f'Đọc lại câu hỏi: {q1}'
        
        formatted_str = f"{q1}\n\n{q2}\n\n"
        
        # Reasoning (if FewShot_CoT)
        if include_reasoning and 'reasoning' in example:
             formatted_str += f"Reasoning:\n{example['reasoning']}\n\n"
        
        # Output
        if include_reasoning:
            output_data = example.get('output', {})
        else:
            output_data = {
                "text": text,
                "opinions": self._simplify_opinions(example.get('opinions', []))
            }

        formatted_str += f"Output:\n{json.dumps(output_data, ensure_ascii=False, indent=2)}"
        
        return formatted_str

    def _format_example_re2_en(self, example: Dict[str, Any], include_reasoning: bool = False) -> str:
        """Format a single example simulating Re-reading logic (English)."""
        text = example.get('text', '')
        
        q1 = f'Analyze the sentiment for the following text: "{text}"'
        q2 = f'Read the question again: {q1}'
        
        formatted_str = f"{q1}\n\n{q2}\n\n"
        
        if include_reasoning and 'reasoning' in example:
             formatted_str += f"Reasoning:\n{example['reasoning']}\n\n"
        
        if include_reasoning:
            output_data = example.get('output', {})
        else:
             output_data = {
                "text": text,
                "opinions": self._simplify_opinions(example.get('opinions', []))
            }

        formatted_str += f"Output:\n{json.dumps(output_data, ensure_ascii=False, indent=2)}"
        
        return formatted_str
        
    def _simplify_opinions(self, opinions: List[Dict]) -> List[Dict]:
        """Simplify opinion format."""
        simplified = []
        for op in opinions:
            simplified_op = {
                "Source": op.get('Source', [[], []])[0],
                "Target": op.get('Target', [[], []])[0],
                "Polar_expression": op.get('Polar_expression', [[], []])[0],
                "Polarity": op.get('Polarity', ''),
                "Intensity": op.get('Intensity', '')
            }
            simplified.append(simplified_op)
        return simplified

    def _build_examples_section_vi(self, examples: List[Dict[str, Any]]) -> str:
        section = "DƯỚI ĐÂY LÀ MỘT SỐ VÍ DỤ MINH HỌA:\n\n"
        include_reasoning = (self.add_method == "FewShot_CoT")
        
        for i, example in enumerate(examples, 1):
            section += f"=== VÍ DỤ {i} ===\n"
            section += self._format_example_re2_vi(example, include_reasoning)
            section += "\n\n"
        return section.strip()

    def _build_examples_section_en(self, examples: List[Dict[str, Any]]) -> str:
        section = "HERE ARE SOME DEMONSTRATION EXAMPLES:\n\n"
        include_reasoning = (self.add_method == "FewShot_CoT")
        
        for i, example in enumerate(examples, 1):
            section += f"=== EXAMPLE {i} ===\n"
            section += self._format_example_re2_en(example, include_reasoning)
            section += "\n\n"
        return section.strip()
    
    def _get_trigger_en(self) -> str:
        """Get the English trigger phrase based on add_method."""
        if self.add_method == "0_CoT":
            return "Let's think step by step."
        elif self.add_method == "PaS":
            return ("Let's first understand the problem and devise a plan to solve the problem. "
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
        """Vietnamese user prompt with Re-reading."""
        
        prompt_parts = []
        
        # 0. Examples (if any)
        if self.n_shot > 0 and self._selected_examples:
            examples_text = self._build_examples_section_vi(self._selected_examples)
            prompt_parts.append(examples_text)
            prompt_parts.append("\nBây giờ, hãy phân tích trường hợp sau:\n")

        # 1. Base Question
        base_question = f"""Phân tích cảm xúc cho văn bản sau (sent_id: {sent_id}):
"{text}"
"""
        # 2. Re-reading instruction
        re_reading_part = f"""Đọc lại câu hỏi: {base_question}
Trả về KẾT QUẢ CHÍNH XÁC theo cấu trúc JSON đã yêu cầu."""
        
        # 3. Add Method Trigger
        trigger = self._get_trigger_vi()
        
        core_prompt = f"{base_question}\n\n{re_reading_part}"
        if trigger:
            core_prompt += f"\n\n{trigger}"
            
        prompt_parts.append(core_prompt)
        
        return "".join(prompt_parts)
    
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
        """English user prompt with Re-reading."""
        
        prompt_parts = []
        
        # 0. Examples (if any)
        if self.n_shot > 0 and self._selected_examples:
            examples_text = self._build_examples_section_en(self._selected_examples)
            prompt_parts.append(examples_text)
            prompt_parts.append("\nNow, analyze the following case:\n")

        # 1. Base Question
        base_question = f"""Analyze the sentiment for the following text (sent_id: {sent_id}):
"{text}"
"""
        # 2. Re-reading instruction
        re_reading_part = f"""Read the question again: {base_question}
Return the EXACT RESULT according to the requested JSON structure."""
        
        # 3. Add Method Trigger
        trigger = self._get_trigger_en()
        
        core_prompt = f"{base_question}\n\n{re_reading_part}"
        if trigger:
            core_prompt += f"\n\n{trigger}"
            
        prompt_parts.append(core_prompt)
        
        return "".join(prompt_parts)


# # Example usage:
# re_reading = ReReadingPrompt(eng=False)
# re_reading.prepare()  # Cache system prompt
# 
# # Use with SentimentDataset
# test_dataset = SentimentDataset(
#     data_path="data/test.json",
#     prompt_generator=re_reading.get_prompt
# )