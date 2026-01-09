# src/demo/utils.py

import spacy
from spacy import displacy
import json
import warnings
from src.utils.postprocessing import extract_position

# Tắt warnings của spaCy
warnings.filterwarnings("ignore")


def get_spacy_model():
    """Load blank spaCy model for visualization container."""
    try:
        nlp = spacy.blank("vi")
    except:
        nlp = spacy.blank("en")
    return nlp


def parse_position_to_tuple(position_str: str) -> tuple:
    """
    Convert position string from extract_position() to tuple.
    
    Args:
        position_str: Position in format "start:end"
    
    Returns:
        Tuple (start, end) or None if invalid
    
    Example:
        >>> parse_position_to_tuple("4:12")
        (4, 12)
    """
    try:
        parts = position_str.split(":")
        return (int(parts[0]), int(parts[1]))
    except:
        return None


def convert_ssa_to_spacy(text, ssa_json):
    """
    Chuyển đổi kết quả SSA JSON thành HTML Dependency Graph.
    
    Uses postprocessing.extract_position() for consistency with main codebase.
    
    Args:
        text: Văn bản gốc.
        ssa_json: Dictionary kết quả SSA (format SemEval).
        
    Returns:
        HTML string để hiển thị.
    """
    nlp = get_spacy_model()
    doc = nlp(text)
    
    options = {
        "compact": False,
        "bg": "#ffffff",
        "distance": 120,
        "color": "#000000",
        "arrow_stroke": 2,
        "arrow_width": 8,
    }

    if isinstance(ssa_json, str):
        try:
            ssa_json = json.loads(ssa_json)
        except:
            return "<div>Invalid JSON Format</div>"

    opinions = ssa_json.get("opinions", [])
    if not opinions:
        return f"<div style='padding: 20px;'>No opinions found in text.</div>"

    words = [token.text for token in doc]
    
    # Tạo mapping từ char index sang token index
    char_to_token = {}
    for token in doc:
        for i in range(token.idx, token.idx + len(token)):
            char_to_token[i] = token.i

    arcs = []
    
    # Duyệt qua các opinions để tạo arrows
    for op in opinions:
        try:
            # Handle format variants
            expr_raw = op.get("Polar_expression", [[None]])[0][0]
            target_raw = op.get("Target", [[None]])[0][0]
            polarity = op.get("Polarity", "Neutral")
            
            if not expr_raw or not target_raw:
                continue

            # Tìm vị trí sử dụng hàm chung từ postprocessing
            expr_pos_str = extract_position(text, expr_raw)
            target_pos_str = extract_position(text, target_raw)
            
            # Convert to tuple
            expr_span = parse_position_to_tuple(expr_pos_str)
            target_span = parse_position_to_tuple(target_pos_str)
            
            if expr_span and target_span and expr_span != (0, 0) and target_span != (0, 0):
                # Map sang token index
                start_token_idx = char_to_token.get(expr_span[0])
                end_token_idx = char_to_token.get(target_span[0])
                
                if start_token_idx is not None and end_token_idx is not None:
                    direction = "left" if start_token_idx > end_token_idx else "right"
                    
                    arcs.append({
                        "start": min(start_token_idx, end_token_idx),
                        "end": max(start_token_idx, end_token_idx),
                        "label": polarity,
                        "dir": direction
                    })
        except Exception as e:
            print(f"Error processing opinion: {e}")
            continue

    # Config manual data
    ex = {
        "words": [{"text": w, "tag": ""} for w in words],
        "arcs": arcs
    }
    
    # Render HTML
    html = displacy.render(ex, style="dep", manual=True, options=options, page=False)
    
    return html