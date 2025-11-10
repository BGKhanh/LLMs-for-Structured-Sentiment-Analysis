# src/prompt_templates/generators.py
from typing import Tuple, Dict, Any

class SingleStagePromptGen:
    def __init__(self, template):
        self.template = template

    def __call__(self, text: str, sent_id: str) -> Tuple[str, str]:
        return self.template.get_prompt(text, sent_id)


class CoTStage1PromptGen:
    def __init__(self, template):
        self.template = template

    def __call__(self, text: str, sent_id: str) -> Tuple[str, str]:
        return self.template.get_prompt(text, sent_id, stage="stage_1")


class CoTStage2PromptGen:
    def __init__(self, template, reasoning_map: Dict[str, Dict[str, Any]]):
        self.template = template
        self.reasoning_map = reasoning_map

    def __call__(self, text: str, sent_id: str) -> Tuple[str, str]:
        sid = str(sent_id)
        r = self.reasoning_map.get(sid, {}).get("raw_response", "")
        if not r:
            raise ValueError(f"No reasoning from Stage 1 for {sid}")
        return self.template.get_prompt(text, sid, stage="stage_2", reasoning=r)