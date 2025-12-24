# src/utils/postprocessing.py

import json
from typing import Dict, Any


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
 
    
def extract_json_from_response(
    raw_response: str,
    model_type: str = "generic"
) -> str:
    """
    Extract JSON from model's raw response (model-agnostic with model-specific preprocessing).
    
    Args:
        raw_response: Raw text from model
        model_type: "gemma", "qwen", or "generic" (for special preprocessing)
    
    Returns:
        Extracted JSON string
    """
    response = raw_response
    
    # === Model-specific preprocessing ===
    if model_type == "gemma":
        # Remove Gemma assistant markers
        for marker in ["assistant:", "assistant", "<assistant>"]:
            if marker in response:
                response = response.split(marker, 1)[1].strip()
    
    elif model_type == "qwen":
        # Remove Qwen thinking tokens
        response = re.sub(r"<think>[\s\S]*?</think>\s*", "", response, flags=re.DOTALL)
        if "<|thought|>" in response:
            response = response.split("<|thought|>")[-1].strip()
    
    # === Generic extraction (works for all models) ===
    # Strategy 1: Extract from ...     
    match = re.search(r'\s*([\s\S]*?)\s*```', response, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Brace matching for first valid JSON
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
                    try:
                        json.loads(candidate)
                        return candidate.strip()
                    except json.JSONDecodeError:
                        break
    
    # Strategy 3: Fallback - clean markdown
    return response.replace('', '').replace('```', '').strip()