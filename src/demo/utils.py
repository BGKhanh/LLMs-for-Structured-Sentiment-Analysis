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
    import re
    
    nlp = get_spacy_model()
    doc = nlp(text)
    
    options = {
        "compact": True,
        "bg": "#ffffff",
        "distance": 100,
        "color": "#111827",
        "arrow_stroke": 2,
        "arrow_width": 8,
        "font": "Inter, Arial, sans-serif",
        "word_spacing": 25,
    }

    if isinstance(ssa_json, str):
        try:
            ssa_json = json.loads(ssa_json)
        except:
            return "<div>Invalid JSON Format</div>"

    opinions = ssa_json.get("opinions", [])
    if not opinions:
        return f"<div style='padding: 20px; font-size: 16px; color: #666;'>No opinions found in text.</div>"

    words = [token.text for token in doc]
    
    # Tạo mapping từ char index sang token index
    char_to_token = {}
    for token in doc:
        for i in range(token.idx, token.idx + len(token)):
            char_to_token[i] = token.i

    arcs = []
    arc_colors = {}  # Map arc index → color
    
    # Duyệt qua các opinions để tạo arrows
    arc_idx = 0
    for op in opinions:
        try:
            expr_raw = op.get("Polar_expression", [[None]])[0][0]
            target_raw = op.get("Target", [[None]])[0][0]
            polarity = op.get("Polarity", "Neutral")
            
            if not expr_raw or not target_raw:
                continue

            expr_pos_str = extract_position(text, expr_raw)
            target_pos_str = extract_position(text, target_raw)
            
            expr_span = parse_position_to_tuple(expr_pos_str)
            target_span = parse_position_to_tuple(target_pos_str)
            
            if expr_span and target_span and expr_span != (0, 0) and target_span != (0, 0):
                start_token_idx = char_to_token.get(expr_span[0])
                end_token_idx = char_to_token.get(target_span[0])
                
                if start_token_idx is not None and end_token_idx is not None:
                    direction = "left" if start_token_idx > end_token_idx else "right"
                    
                    # Determine color
                    if polarity == "Positive":
                        color = "#22c55e"
                    elif polarity == "Negative":
                        color = "#ef4444"
                    else:
                        color = "#6b7280"
                    
                    arcs.append({
                        "start": min(start_token_idx, end_token_idx),
                        "end": max(start_token_idx, end_token_idx),
                        "label": polarity,
                        "dir": direction
                    })
                    
                    arc_colors[arc_idx] = color
                    arc_idx += 1
                    
        except Exception as e:
            print(f"⚠️ Error processing opinion: {e}")
            continue

    if not arcs:
        return f"""<div style='padding: 20px; font-size: 16px; color: #666;'>
            <p>Opinions found but could not visualize relationships.</p>
            <p>Total opinions: {len(opinions)}</p>
        </div>"""

    ex = {
        "words": [{"text": w, "tag": ""} for w in words],
        "arcs": arcs
    }
    
    # Render HTML
    html = displacy.render(ex, style="dep", manual=True, options=options, page=False)
    
    # CRITICAL: Post-process HTML để inject màu trực tiếp vào SVG paths
    # displaCy generates paths in order, so we can map by index
    
    # Find all <g class="displacy-arrow"> blocks
    pattern = r'(<g class="displacy-arrow">.*?</g>)'
    arrow_blocks = re.findall(pattern, html, re.DOTALL)
    
    # Replace each arrow block with colored version
    for idx, block in enumerate(arrow_blocks):
        if idx in arc_colors:
            color = arc_colors[idx]
            
            # Replace stroke color in path
            new_block = re.sub(
                r'stroke="[^"]*"',
                f'stroke="{color}" stroke-width="3"',
                block
            )
            
            # Replace marker color
            new_block = re.sub(
                r'fill="[^"]*"',
                f'fill="{color}"',
                new_block
            )
            
            # Replace label text color
            new_block = re.sub(
                r'(<text[^>]*class="displacy-label"[^>]*)(>)',
                rf'\1 fill="{color}" font-weight="bold" font-size="14"\2',
                new_block
            )
            
            html = html.replace(block, new_block)
    
    # Wrap with scrollable container
    enhanced_html = f"""
    <div style="width: 100%; overflow-x: auto; overflow-y: hidden; padding: 10px 0;">
        <style>
            .displacy-container {{
                min-width: max-content !important;
                padding: 30px 20px !important;
                background: #ffffff !important;
                border-radius: 8px !important;
                border: 1px solid #e5e7eb !important;
            }}
            
            .displacy-word {{
                font-size: 16px !important;
                font-weight: 600 !important;
                fill: #111827 !important;
            }}
            
            .displacy-label {{
                font-size: 14px !important;
                font-weight: 700 !important;
            }}
        </style>
        {html}
    </div>
    """
    
    return enhanced_html