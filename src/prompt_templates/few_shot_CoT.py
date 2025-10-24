# src/prompt_templates/few_shot_CoT.py

from typing import Dict, Any, List, Tuple
from .base import BasePromptTemplate
import json

class FewShotCoTPrompt(BasePromptTemplate):
    """
    Few-shot Chain-of-Thought prompt generator.
    
    Combines few-shot learning with chain-of-thought reasoning.
    Uses hardcoded examples with pre-written reasoning demonstrations.
    
    Paper: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
    Link: https://arxiv.org/abs/2201.11903
    
    Design:
    - Hardcoded examples pool with reasoning demonstrations
    - n_shot parameter selects how many examples to include
    - Examples are curated manually for quality reasoning
    - Single stage (no separate reasoning extraction like zero-shot CoT)
    """
    
    # Hardcoded examples pool with reasoning
    EXAMPLES_POOL_VI = [
        {
            "text": "quay quay cái lồn , thấy bị bắt nạt thì ra nói một câu bảo vệ người ta , có khi tối về lại có người nằm ôm . cứ cầm điện thoại pin với chả không pin",
            "reasoning": """
Bước 1: Xác định các Biểu thức Cảm xúc
Phát hiện 5 polar expressions: "quay quay cái lồn", "thấy bị bắt nạt", "nói một câu bảo vệ người ta", "tối về lại có người nằm ôm", "cứ cầm điện thoại pin với chả không pin".

Bước 2: Phân tích Tuần tự từng Opinion
--- Opinion 1: "quay quay cái lồn" ---

Source: Không có vì không xuất hiện đại từ nhân xưng nào như "tôi", "mình".
Target: Không có vì không có đại từ xưng hô "mày", "cậu" hay danh từ cụ thể.
Polar_expression: "quay quay cái lồn" vì đây là cụm từ thô tục chỉ trích hành động quay phim.
Polarity: Negative vì sử dụng từ ngữ xúc phạm để chê bai.
Intensity: Standard vì mức độ thô tục ở mức trung bình, không cực đoan.

--- Opinion 2: "thấy bị bắt nạt" ---

Source: Không có.
Target: Không có.
Polar_expression: "thấy bị bắt nạt" vì mô tả tình huống tiêu cực.
Polarity: Negative vì bắt nạt là hành vi có hại.
Intensity: Standard vì diễn đạt ở mức độ thông thường.

--- Opinion 3: "nói một câu bảo vệ người ta" ---

Source: Không có.
Target: Không có, "người ta" ở đây là đại từ chung chung không xác định.
Polar_expression: "nói một câu bảo vệ người ta" vì mô tả hành động tích cực.
Polarity: Positive vì bảo vệ là hành vi tốt đẹp.
Intensity: Standard vì là hành động thông thường, không quá mạnh mẽ.

--- Opinion 4: "tối về lại có người nằm ôm" ---

Source: Không có.
Target: Không có.
Polar_expression: "tối về lại có người nằm ôm" vì ám chỉ kết quả tình cảm tích cực.
Polarity: Positive vì thể hiện sự gần gũi, thân mật.
Intensity: Standard vì diễn đạt ở mức độ bình thường.

--- Opinion 5: "cứ cầm điện thoại pin với chả không pin" ---

Source: Không có.
Target: Không có.
Polar_expression: "cứ cầm điện thoại pin với chả không pin" vì mô tả thói quen quay phim liên tục.
Polarity: Neutral vì chỉ nhận xét hành vi mà không có tính xúc phạm hay khen ngợi rõ ràng.
Intensity: Standard vì diễn đạt bình thường.
""",
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
            "reasoning": """
Bước 1: Xác định các Biểu thức Cảm xúc
Phát hiện 3 polar expressions: "yêu sắp 2 năm rồi", "không biết người yêu kể chuyện cho nghe là gì", "đòi mấy lần toàn kêu không bình thường xịu xịu".

Bước 2: Phân tích Tuần tự từng Opinion

--- Opinion 1: "yêu sắp 2 năm rồi" ---
Source: Không có vì không xuất hiện đại từ nhân xưng rõ ràng.
Target: Không có vì không có đối tượng cụ thể được nhắc đến.
Polar_expression: "yêu sắp 2 năm rồi" vì thể hiện mối quan hệ tình cảm lâu dài.
Polarity: Positive vì việc yêu nhau gần 2 năm thể hiện tình cảm ổn định, bền vững.
Intensity: Standard vì diễn đạt ở mức độ bình thường, không quá phóng đại.

--- Opinion 2: "không biết người yêu kể chuyện cho nghe là gì" ---
Source: Không có.
Target: Không có, "người yêu" ở đây là danh từ chung chỉ đối tượng nhưng không phải target cụ thể trong ngữ cảnh phân tích.
Polar_expression: "không biết người yêu kể chuyện cho nghe là gì" vì bày tỏ sự thiếu vắng giao tiếp, cảm giác buồn bã (kèm emoji 😢).
Polarity: Negative vì thể hiện sự thất vọng, thiếu kết nối trong quan hệ.
Intensity: Standard vì diễn đạt ở mức độ thông thường, không quá gay gắt.

--- Opinion 3: "đòi mấy lần toàn kêu không bình thường xịu xịu" ---
Source: Không có.
Target: Không có.
Polar_expression: "đòi mấy lần toàn kêu không bình thường xịu xịu" vì mô tả phản ứng từ chối của người yêu.
Polarity: Neutral vì chỉ khách quan mô tả phản ứng mà không mang tính xúc phạm hay khen ngợi rõ ràng, "xịu xịu" có thể hiểu là nhạt nhẽo nhưng không đủ mạnh để đánh giá tiêu cực.
Intensity: Standard vì diễn đạt bình thường.
""",
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
            "reasoning": """
Bước 1: Xác định các Biểu thức Cảm xúc
Phát hiện 1 polar expression: "giúp các bạn có thêm niềm tin trong cuộc sống".

Bước 2: Phân tích Tuần tự từng Opinion

--- Opinion 1: "giúp các bạn có thêm niềm tin trong cuộc sống" ---
Source: Không có vì không xuất hiện đại từ nhân xưng rõ ràng chỉ người phát biểu.
Target: "câu chuyện admin vừa bịa ra" vì đây là đối tượng mà cảm xúc hướng tới, là sự vật được nhắc đến trong câu.
Polar_expression: "giúp các bạn có thêm niềm tin trong cuộc sống" vì thể hiện tác động tích cực, mang lại hy vọng.
Polarity: Positive vì "có thêm niềm tin" và "giúp đỡ" đều là những yếu tố tích cực, mặc dù có emoji 😂 mang tính châm biếm nhưng bản thân cụm từ vẫn mang nghĩa tích cực.
Intensity: Standard vì diễn đạt ở mức độ bình thường, không quá phóng đại hay quá nhẹ nhàng.
""",
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
        # Add more examples as needed
    ]
    
    EXAMPLES_POOL_EN = [
        {
            "text": "quay quay cái lồn , thấy bị bắt nạt thì ra nói một câu bảo vệ người ta , có khi tối về lại có người nằm ôm . cứ cầm điện thoại pin với chả không pin",
            "reasoning": """
Step 1: Identify Polar Expressions
Detected 5 polar expressions: "quay quay cái lồn" (fucking filming), "thấy bị bắt nạt" (seeing someone bullied), "nói một câu bảo vệ người ta" (saying a word to protect them), "tối về lại có người nằm ôm" (going home tonight and having someone to hug), "cứ cầm điện thoại pin với chả không pin" (just holding the phone, battery or no battery).
Step 2: Sequential Analysis of Each Opinion
--- Opinion 1: "quay quay cái lồn" ---

Source: None, as no personal pronouns like "tôi" (I), "mình" (I/we) appear.
Target: None, as there are no second-person pronouns "mày" (you), "cậu" (you) or specific nouns.
Polar_expression: "quay quay cái lồn" (fucking filming) because it is a vulgar phrase criticizing the act of filming.
Polarity: Negative because it uses insulting language to criticize.
Intensity: Standard because the level of vulgarity is moderate, not extreme.

--- Opinion 2: "thấy bị bắt nạt" ---

Source: None.
Target: None.
Polar_expression: "thấy bị bắt nạt" (seeing someone bullied) because it describes a negative situation.
Polarity: Negative because bullying is a harmful behavior.
Intensity: Standard because the expression is at a normal level.

--- Opinion 3: "nói một câu bảo vệ người ta" ---

Source: None.
Target: None, "người ta" (them/other people) here is an indeterminate general pronoun.
Polar_expression: "nói một câu bảo vệ người ta" (saying a word to protect them) because it describes a positive action.
Polarity: Positive because protecting is a good deed.
Intensity: Standard because it is a normal action, not too strong.

--- Opinion 4: "tối về lại có người nằm ôm" ---

Source: None.
Target: None.
Polar_expression: "tối về lại có người nằm ôm" (going home tonight and having someone to hug) because it implies a positive emotional result.
Polarity: Positive because it shows closeness and intimacy.
Intensity: Standard because the expression is at a normal level.

--- Opinion 5: "cứ cầm điện thoại pin với chả không pin" ---

Source: None.
Target: None.
Polar_expression: "cứ cầm điện thoại pin với chả không pin" (just holding the phone, battery or no battery) because it describes the habit of filming continuously.
Polarity: Neutral because it only comments on the behavior without clear insult or praise.
Intensity: Standard because the expression is normal.
""",
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
            "reasoning": """
Step 1: Identify Polar Expressions
Detected 3 polar expressions: "yêu sắp 2 năm rồi" (been in love for almost 2 years), "không biết người yêu kể chuyện cho nghe là gì" (don't know what it's like to have a lover tell stories), "đòi mấy lần toàn kêu không bình thường xịu xịu" (asking several times, they just say it's not normal, looking dejected/sullen).

Step 2: Sequential Analysis of Each Opinion

--- Opinion 1: "yêu sắp 2 năm rồi" ---
Source: None, as no clear personal pronouns appear.
Target: None, as no specific target is mentioned.
Polar_expression: "yêu sắp 2 năm rồi" (been in love for almost 2 years) as it expresses a long-term romantic relationship.
Polarity: Positive because being in love for almost 2 years shows stable, lasting affection.
Intensity: Standard because the expression is at a normal level, not overly exaggerated.

--- Opinion 2: "không biết người yêu kể chuyện cho nghe là gì" ---
Source: None.
Target: None, "người yêu" (lover/boyfriend/girlfriend) here is a general noun referring to the subject but not a specific target in the analysis context.
Polar_expression: "không biết người yêu kể chuyện cho nghe là gì" (don't know what it's like to have a lover tell stories) as it expresses a lack of communication, a feeling of sadness (accompanied by the 😢 emoji).
Polarity: Negative because it shows disappointment, a lack of connection in the relationship.
Intensity: Standard because the expression is at a normal level, not too harsh.

--- Opinion 3: "đòi mấy lần toàn kêu không bình thường xịu xịu" ---
Source: None.
Target: None.
Polar_expression: "đòi mấy lần toàn kêu không bình thường xịu xịu" (asking several times, they just say it's not normal, looking dejected/sullen) as it describes the lover's refusal/negative reaction.
Polarity: Neutral because it objectively describes the reaction without clear insult or praise; "xịu xịu" (dejected/sullen/flat) can be understood as dull but not strong enough to be rated negative.
Intensity: Standard because the expression is normal.
""",
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
            "reasoning": """
Step 1: Identify Polar Expressions
Detected 1 polar expression: "giúp các bạn có thêm niềm tin trong cuộc sống" (helps you have more faith in life).

Step 2: Sequential Analysis of Each Opinion

--- Opinion 1: "giúp các bạn có thêm niềm tin trong cuộc sống" ---
Source: None, as no clear personal pronoun explicitly identifies the speaker.
Target: "câu chuyện admin vừa bịa ra" (the story the admin just fabricated) because this is the object the sentiment is directed at, the entity mentioned in the sentence.
Polar_expression: "giúp các bạn có thêm niềm tin trong cuộc sống" (helps you have more faith in life) as it expresses a positive effect, bringing hope.
Polarity: Positive because "have more faith" and "help" are positive elements, although the 😂 emoji suggests sarcasm, the phrase itself still carries a positive meaning.
Intensity: Standard because the expression is at a normal level, neither overly exaggerated nor too mild.
""",
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
    ]
    
    def __init__(self, eng: bool = False, n_shot: int = 3):
        """
        Initialize Few-shot CoT prompt generator.
        
        Args:
            eng: If True, use English prompts and examples. 
                 If False, use Vietnamese.
            n_shot: Number of examples to include (default: 3).
                    Must be <= number of examples in pool.
        
        Note:
            Examples are hardcoded in EXAMPLES_POOL_VI and EXAMPLES_POOL_EN.
            These examples should be carefully curated with high-quality reasoning.
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
        """
        Select n_shot examples from hardcoded pool.
        
        Note:
            Takes first n_shot examples from pool.
            Can be modified to random selection if needed.
        """
        examples_pool = self.EXAMPLES_POOL_EN if self.eng else self.EXAMPLES_POOL_VI
        
        if self.n_shot > len(examples_pool):
            raise ValueError(
                f"Requested n_shot={self.n_shot} but only "
                f"{len(examples_pool)} examples available in pool"
            )
        
        # Take first n examples (can use random if needed)
        self._selected_examples = examples_pool[:self.n_shot]
        print(f"✅ Selected {self.n_shot} hardcoded examples for few-shot CoT")
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt with component definitions and examples.
        
        Returns:
            System prompt including few-shot CoT demonstrations
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
        
        Args:
            text: Text to analyze
            sent_id: Sample identifier
        
        Returns:
            User prompt asking for analysis of current text
        """
        if self.eng:
            return self._get_user_prompt_en(text, sent_id)
        else:
            return self._get_user_prompt_vi(text, sent_id)
    
    # ==================== VIETNAMESE PROMPTS ====================
    
    def _get_system_prompt_vi(self) -> str:
        """Vietnamese system prompt with task definition."""
        base_prompt = """Bạn là một chuyên gia phân tích cảm xúc tiếng Việt. 
Nhiệm vụ của bạn là phân tích các bình luận, trình bày quá trình suy luận của mình, và cuối cùng cung cấp kết quả dưới dạng một khối JSON duy nhất. 
Hãy làm theo các ví dụ được cung cấp.
"""
        
        # Add few-shot examples if available
        if self.n_shot > 0 and self._selected_examples:
            examples_section = self._build_examples_section_vi(self._selected_examples)
            base_prompt += f"\n\n{examples_section}"
        
        return base_prompt
    
    def _get_user_prompt_vi(self, text: str, sent_id: str) -> str:
        """Vietnamese user prompt for current sample."""
        return f"""
Bây giờ, hãy phân tích văn bản sau (sent_id: {sent_id}):
"{text}"

Hãy suy luận từng bước một, sau đó trả về JSON cuối cùng."""
    
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
        """English system prompt with task definition."""
        base_prompt = """You are a Vietnamese sentiment analysis expert.
Your task is to analyze the comments, present your reasoning process, and finally provide the result in a single JSON block. 
Please follow the examples provided.
"""
        
        # Add few-shot examples if available
        if self.n_shot > 0 and self._selected_examples:
            examples_section = self._build_examples_section_en(self._selected_examples)
            base_prompt += f"\n\n{examples_section}"
        
        return base_prompt
    
    def _get_user_prompt_en(self, text: str, sent_id: str) -> str:
        """English user prompt for current sample."""
        return f"""
Now, analyze the following text (sent_id: {sent_id}):
"{text}"

Think step by step, then return the final JSON."""
    
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


# # Example usage:
# few_shot_cot = FewShotCoTPrompt(eng=False, n_shot=3)
# few_shot_cot.prepare()  # Select examples, cache system prompt
# 
# # Use with SentimentDataset
# test_dataset = SentimentDataset(
#     data_path="data/test.json",
#     prompt_generator=few_shot_cot.get_prompt
# )