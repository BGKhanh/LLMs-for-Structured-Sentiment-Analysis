from typing import Dict, Any, List, Tuple, Optional
import random
from .base import BasePromptTemplate
import json

class FewShotPrompt(BasePromptTemplate):
    """
    Few-shot prompt generator for structured sentiment analysis.
    
    Automatically injects n examples from dataset into prompt to guide model.
    Examples are guaranteed to be different from the current inference sample.
    """
    
    def __init__(
        self,
        eng: bool = False,
        n_shot: int = 0,
        examples_pool: Optional[Dataset] = None,
        examples_pool_path: Optional[str] = None
    ):
        """
        Initialize Few-shot prompt generator.
        
        Args:
            eng: If True, use English prompts. If False, use Vietnamese prompts.
            n_shot: Number of examples to include in prompt. If 0, acts as zero-shot.
            examples_pool: Dataset object containing example pool (e.g., SentimentDataset).
                          Takes priority over examples_pool_path.
            examples_pool_path: Path to JSON file containing example pool (e.g., train.json).
                               Used only if examples_pool is None.
        
        Note:
            Priority: examples_pool > examples_pool_path.
            At least one of them should be provided if n_shot > 0.
        """
        super().__init__(eng)
        
        if n_shot < 0:
            raise ValueError("n_shot must be >= 0")
        
        self.n_shot = n_shot
        self.examples_pool_path = examples_pool_path
        self._examples_pool_raw = examples_pool  # Raw list provided by user
        
        # Lazy loaded in prepare()
        self._examples_pool = None
        self._selected_examples = None  # Fixed examples for all samples
        
    def prepare(self) -> None:
        """Load examples pool and select fixed examples."""
        if self._is_prepared:
            return
        
        # Load examples if needed
        if self.n_shot > 0:
            self._load_examples_pool()
            self._select_fixed_examples()
        
        # Parent class will build and cache system prompt
        super().prepare()
        
    def _select_fixed_examples(self) -> None:
        """
        Select fixed examples for all samples (ensures consistency and fairness).
        
        Note:
            Same examples are used for all inference samples.
            This ensures fair comparison and better efficiency.
        """
        if len(self._examples_pool) < self.n_shot:
            raise ValueError(
                f"Not enough examples. Requested {self.n_shot}, "
                f"but only {len(self._examples_pool)} available."
            )
        
        self._selected_examples = random.sample(self._examples_pool, self.n_shot)
        print(f"✅ Selected {self.n_shot} fixed examples for few-shot prompting")
    
    def _load_examples_pool(self) -> None:
        """
        Load examples from provided source.
        
        Priority: _examples_pool_raw > examples_pool_path
        """
        # Priority 1: Raw list
        if self._examples_pool_raw is not None:
            self._examples_pool = self._examples_pool_raw
            print(f"✅ Loaded {len(self._examples_pool)} examples from provided list")
            return
        
        # Priority 2: File path
        if self.examples_pool_path is not None:
            try:
                with open(self.examples_pool_path, 'r', encoding='utf-8') as f:
                    self._examples_pool = json.load(f)
                print(f"✅ Loaded {len(self._examples_pool)} examples from {self.examples_pool_path}")
                return
            except FileNotFoundError:
                raise FileNotFoundError(f"Examples pool file not found: {self.examples_pool_path}")
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format in file: {self.examples_pool_path}")
        
        # No source provided
        raise ValueError(
            f"n_shot={self.n_shot} but no examples pool available. "
            "Provide examples_pool or examples_pool_path parameter."
        )
        
    def _build_system_prompt(self) -> str:
        """
        Build system prompt (cached, same for all samples).
        
        Returns:
            System prompt with component definitions and format instructions.
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
        Build user prompt with examples (varies per sample only by text/sent_id).
        
        Args:
            text: Text to analyze
            sent_id: Sample identifier
        
        Returns:
            User prompt with examples (if any) and current sample
        """
        if self.eng:
            return self._get_user_prompt_en(text, sent_id)
        else:
            return self._get_user_prompt_vi(text, sent_id)
    
    def _get_system_prompt_vi(self) -> str:
        """Vietnamese system prompt."""
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

QUAN TRỌNG: CHỈ TRẢ VỀ JSON, KHÔNG GIẢI THÍCH THÊM, KHÔNG TRÌNH BÀY QUÁ TRÌNH SUY LUẬN.
"""
    
    def _get_user_prompt_vi(self, text: str, sent_id: str) -> str:
        """Vietnamese user prompt with examples."""
        # Build examples section if few-shot
        if self.n_shot > 0 and self._selected_examples:
            examples_text = self._build_examples_section_vi(self._selected_examples)
            return f"""{examples_text}

Bây giờ, phân tích cảm xúc cho văn bản sau (sent_id: {sent_id}):
Input: "{text}"
Output:
"""
        else:
            # Zero-shot case
            return f"""Phân tích cảm xúc cho văn bản sau (sent_id: {sent_id}):
"{text}"

Trả về KẾT QUẢ CHÍNH XÁC theo cấu trúc JSON đã yêu cầu.
"""
    
    def _get_system_prompt_en(self) -> str:
        """English system prompt."""
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

IMPORTANT: RETURN ONLY THE JSON, PROVIDE NO FURTHER EXPLANATION, DO NOT SHOW THE REASONING PROCESS.
"""
    
    def _get_user_prompt_en(self, text: str, sent_id: str) -> str:
        """English user prompt with examples."""
        # Build examples section if few-shot
        if self.n_shot > 0 and self._selected_examples:
            examples_text = self._build_examples_section_en(self._selected_examples)
            return f"""{examples_text}

Now, analyze the sentiment for the following text (sent_id: {sent_id}):
Input: "{text}"
Output:
"""
        else:
            # Zero-shot case
            return f"""Analyze the sentiment for the following text (sent_id: {sent_id}):
"{text}"

Return the EXACT RESULT according to the requested JSON structure.
"""
    
    
    def _format_example(self, example: Dict[str, Any]) -> str:
        """
        Format a single example for inclusion in prompt.
        
        Args:
            example: Example dictionary with 'text' and 'opinions'
            
        Returns:
            Formatted example string
            
        Note:
            This converts the example into a readable format showing
            input text and expected JSON output.
        """

        
        formatted_output = {
            "text": example.get('text', ''),
            "opinions": self._simplify_opinions(example.get('opinions', []))
        }
        
        formatted_str = f"""Input: "{example.get('text', '')}"
Output: {json.dumps(formatted_output, ensure_ascii=False, indent=2)}"""
        
        return formatted_str
    
    def _simplify_opinions(self, opinions: List[Dict]) -> List[Dict]:
        """
        Simplify opinion format for demonstration (remove character indices).
        
        Args:
            opinions: List of opinion dictionaries
            
        Returns:
            Simplified opinions (only text spans, no indices)
        """
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
    
    
    def _build_examples_section_vi(
        self,
        examples: List[Dict[str, Any]]
    ) -> str:
        """
        Build Vietnamese examples section for prompt.
        
        Args:
            examples: List of example dictionaries
            
        Returns:
            Formatted examples section
        """
        section = "Dưới đây là một số ví dụ:\n\n"
        
        for i, example in enumerate(examples, 1):
            section += f"Ví dụ {i}:\n"
            section += self._format_example(example)
            section += "\n\n"
        
        return section.strip()
    
    def _build_examples_section_en(
        self,
        examples: List[Dict[str, Any]]
    ) -> str:
        """
        Build English examples section for prompt.
        
        Args:
            examples: List of example dictionaries
            
        Returns:
            Formatted examples section
        """
        section = "Here are some examples:\n\n"
        
        for i, example in enumerate(examples, 1):
            section += f"Example {i}:\n"
            section += self._format_example(example)
            section += "\n\n"
        
        return section.strip()


# # Option 1: File path (đơn giản nhất)
# few_shot = FewShotPrompt(
#     eng=False,
#     n_shot=3,
#     examples_pool_path="data/train.json"
# )
# few_shot.prepare()  # Load & select examples, cache system prompt

# # Option 2: Raw list (advanced)
# import json
# with open("data/train.json") as f:
#     train_data = json.load(f)

# few_shot = FewShotPrompt(
#     eng=False,
#     n_shot=3,
#     examples_pool=train_data  # Direct list
# )
# few_shot.prepare()

# # Use with SentimentDataset
# test_dataset = SentimentDataset(
#     data_path="data/test.json",
#     prompt_generator=few_shot.get_prompt  # Simple interface!
# )