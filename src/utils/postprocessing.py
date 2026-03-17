# src/utils/postprocessing.py

import json
import re
from typing import Any


def extract_position(text: str, expression: str) -> str:
    """
    Extract start and end positions of expression in text.
    
    Args:
        text: Original text
        expression: Expression to find position for
        
    Returns:
        Position string in format "start:end" (0-indexed)
        
    Example:
        >>> extract_position("Tôi rất vui", "rất vui")
        "4:12"
    
    Note:
        Model-agnostic utility for SemEval format compliance.
    """
    start = text.find(expression)
    if start == -1:
        return "0:0"
    end = start + len(expression)
    return f"{start}:{end}"

def _extract_by_fields(text: str) -> str:
    """Last resort: build JSON bằng regex extraction từng field."""
    sent_id_m = re.search(r'"sent_id"\s*:\s*(\d+)', text)
    text_m = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    result = {
        "sent_id": int(sent_id_m.group(1)) if sent_id_m else None,
        "text": text_m.group(1) if text_m else "",
        "opinions": []
    }
    opinions_start = text.find('"opinions"')
    if opinions_start != -1:
        arr_start = text.find('[', opinions_start)
        if arr_start != -1:
            depth, obj_start = 0, None
            for i in range(arr_start, len(text)):
                if text[i] == '{':
                    if depth == 0:
                        obj_start = i
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0 and obj_start is not None:
                        obj_text = text[obj_start:i+1]
                        try:
                            result["opinions"].append(json.loads(obj_text))
                        except Exception:
                            pol_m = re.search(r'"Polarity"\s*:\s*"(\w+)"', obj_text)
                            int_m = re.search(r'"Intensity"\s*:\s*"(\w+)"', obj_text)
                            if pol_m:
                                result["opinions"].append({
                                    "Source": [], "Target": [], "Polar_expression": [],
                                    "Polarity": pol_m.group(1),
                                    "Intensity": int_m.group(1) if int_m else ""
                                })
                        obj_start = None
    return json.dumps(result, ensure_ascii=False)

def extract_json_from_response(raw_response: str) -> str:
    """
    Extract JSON từ raw response của LLM với xử lý robust.

    Các lỗi được fix:
    1. Bug regex cũ (Strategy 1): dùng ký tự invisible thay vì backtick thật
    2. Invalid/over-escaped sequences: \\\\\\\" -> \\"
    3. Curly/unicode quotes: " " → straight quotes
    4. Triple-double-quotes bao quanh text field: \"\"\"...\"\"\"
    5. Trailing extra quote: \"\"  ,  ->  \" ,
    6. Unclosed Polar_expression array
    7. Fallback field-by-field extraction
    """
    response = raw_response.strip()

    # Bước 1: Extract block giữa ```json...``` (fix bug backtick invisible)
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if match:
        candidate = match.group(1).strip()
    else:
        candidate = ""
        first_brace = response.find('{')
        if first_brace != -1:
            open_braces = 0
            for i in range(first_brace, len(response)):
                if response[i] == '{':
                    open_braces += 1
                elif response[i] == '}':
                    open_braces -= 1
                    if open_braces == 0:
                        candidate = response[first_brace:i+1]
                        break
        if not candidate:
            return response.replace('```', '').strip()

    # Bước 2: Parse trực tiếp
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass

    # Bước 3: Fix lần lượt các lỗi phổ biến
    fixed = candidate

    # Fix curly/unicode quotes
    fixed = fixed.replace('\u201c', '\\"').replace('\u201d', '\\"')
    fixed = fixed.replace('\u2018', "\\'").replace('\u2019', "\\'")

    # Fix over-escaped: 3+ backslashes trước quote -> \"
    fixed = re.sub(r'\\{3,}"', '\\\\"', fixed)

    # Fix triple-double-quotes: "text": """..."""
    def fix_triple_quotes(m):
        inner = m.group(1).replace('"', '\\"')
        return f'"text": "{inner}"'
    fixed = re.sub(r'"text":\s*"""([\s\S]*?)"""', fix_triple_quotes, fixed)

    # Fix double outer quotes: "text": ""...""
    fixed = re.sub(
        r'"text":\s*""([\s\S]*?)""(?=\s*[,}])',
        lambda m: f'"text": "{m.group(1).replace(chr(34), chr(92)+chr(34))}"',
        fixed
    )

    # Fix trailing extra quote: ""  ,  ->  " ,
    fixed = re.sub(r'""\s*([,}\]])', r'"\1', fixed)

    # Fix unclosed Polar_expression array
    fixed = re.sub(r'(\["[^"\]]*")\s*\n(\s*"Polarity")', r'\1]\n\2', fixed)

    try:
        json.loads(fixed)
        return fixed
    except json.JSONDecodeError:
        pass

    # Bước 4: Last resort - field-by-field extraction
    return _extract_by_fields(candidate)

def postprocess_response(
    response_text: str,
    original_text: str,
    sent_id: Any
) -> str:
    """
    Normalize model response to SemEval format.
    
    Args:
        response_text: JSON response from model (after extract_response())
        original_text: Original input text
        sent_id: Sentence identifier
        
    Returns:
        Formatted JSON string with validated structure and positions
        
    Note:
        - Works with all models (model-agnostic)
        - Handles multiple opinion formats
        - Auto-extracts positions for all text spans
        - Validates Polarity and Intensity values
    
    Example:
        >>> json_str = model.extract_response(raw_response)
        >>> final = postprocess_response(json_str, original_text, sent_id)
    """
    try:
        result = json.loads(response_text)
        result["sent_id"] = sent_id
        result["text"] = original_text
        
        if "opinions" in result and isinstance(result["opinions"], list):
            for opinion in result["opinions"]:
                if not isinstance(opinion, dict):
                    continue
                
                # Process each component: Source, Target, Polar_expression
                for component in ["Source", "Target", "Polar_expression"]:
                    if component not in opinion:
                        opinion[component] = [[], []]
                        continue
                    
                    component_data = opinion[component]
                    
                    # Handle different formats
                    if isinstance(component_data, list):
                        # Check if already in [texts, positions] format
                        if (len(component_data) == 2 and 
                            isinstance(component_data[0], list) and 
                            isinstance(component_data[1], list)):
                            # Old format - re-extract positions for accuracy
                            texts = component_data[0]
                        else:
                            # New format - just array of text spans
                            texts = component_data
                        
                        # Clean and extract positions
                        valid_texts = [
                            text for text in texts 
                            if isinstance(text, str) and text.strip()
                        ]
                        positions = [
                            extract_position(original_text, text) 
                            for text in valid_texts
                        ]
                        opinion[component] = [valid_texts, positions]
                        
                    elif isinstance(component_data, str) and component_data.strip():
                        # Single string
                        text = component_data.strip()
                        position = extract_position(original_text, text)
                        opinion[component] = [[text], [position]]
                    else:
                        # Empty or invalid
                        opinion[component] = [[], []]
                
                # Validate Polarity
                valid_polarities = ["Positive", "Negative", "Neutral"]
                if ("Polarity" not in opinion or 
                    opinion["Polarity"] not in valid_polarities):
                    opinion["Polarity"] = ""
                    
                # Validate Intensity
                valid_intensities = ["Strong", "Standard", "Weak"]
                if ("Intensity" not in opinion or 
                    opinion["Intensity"] not in valid_intensities):
                    opinion["Intensity"] = ""
        else:
            result["opinions"] = []
            
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except json.JSONDecodeError:
        # Fallback: return empty structure
        default_result = {
            "sent_id": sent_id,
            "text": original_text,
            "opinions": []
        }
        return json.dumps(default_result, ensure_ascii=False, indent=2)
 
    
def postprocess_response(response_text: str, original_text: str, sent_id: Any) -> str:
    """Normalize model response to SemEval format."""
    try:
        result = json.loads(response_text)
        result["sent_id"] = sent_id
        result["text"] = original_text

        if "opinions" in result and isinstance(result["opinions"], list):
            for opinion in result["opinions"]:
                if not isinstance(opinion, dict):
                    continue

                for component in ["Source", "Target", "Polar_expression"]:
                    if component not in opinion:
                        opinion[component] = [[], []]
                        continue
                    component_data = opinion[component]
                    if isinstance(component_data, list):
                        if (len(component_data) == 2 and
                                isinstance(component_data[0], list) and
                                isinstance(component_data[1], list)):
                            texts = component_data[0]
                        else:
                            texts = component_data
                        valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
                        positions = [extract_position(original_text, t) for t in valid_texts]
                        opinion[component] = [valid_texts, positions]
                    elif isinstance(component_data, str) and component_data.strip():
                        t = component_data.strip()
                        opinion[component] = [[t], [extract_position(original_text, t)]]
                    else:
                        opinion[component] = [[], []]

                valid_polarities = ["Positive", "Negative", "Neutral"]
                if "Polarity" not in opinion or opinion["Polarity"] not in valid_polarities:
                    opinion["Polarity"] = ""

                valid_intensities = ["Strong", "Standard", "Weak"]
                if "Intensity" not in opinion or opinion["Intensity"] not in valid_intensities:
                    opinion["Intensity"] = ""
        else:
            result["opinions"] = []

        return json.dumps(result, ensure_ascii=False, indent=2)

    except json.JSONDecodeError:
        return json.dumps({"sent_id": sent_id, "text": original_text, "opinions": []},
                          ensure_ascii=False, indent=2)