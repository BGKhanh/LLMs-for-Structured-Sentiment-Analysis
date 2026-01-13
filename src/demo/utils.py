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
    
    # FIXED: Màu sắc và layout rõ ràng
    options = {
        "compact": True,  # ← Compact mode để text không quá dài
        "bg": "#f9fafb",
        "distance": 100,  # ← Giảm distance để gọn hơn
        "color": "#111827",
        "arrow_stroke": 2,
        "arrow_width": 8,
        "font": "Inter, Arial, sans-serif",
        "word_spacing": 25,  # ← Spacing giữa các từ
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
                        "dir": direction,
                        "color": "#22c55e" if polarity == "Positive" else "#ef4444" if polarity == "Negative" else "#6b7280"  # ← Direct color
                    })
        except Exception as e:
            print(f"⚠️ Error processing opinion: {e}")
            continue

    if not arcs:
        return f"""<div style='padding: 20px; font-size: 16px; color: #666;'>
            <p>Opinions found but could not visualize relationships.</p>
            <p>Total opinions: {len(opinions)}</p>
        </div>"""

    # Config manual data
    ex = {
        "words": [{"text": w, "tag": ""} for w in words],
        "arcs": arcs
    }
    
    # Render HTML
    html = displacy.render(ex, style="dep", manual=True, options=options, page=False)
    
    # CRITICAL FIX: Thêm CSS với !important để override displaCy defaults
    enhanced_html = f"""
    <div style="width: 100%; overflow-x: auto; overflow-y: hidden; padding: 10px 0;">
        <style>
            /* Container với scroll ngang */
            .displacy-container {{
                min-width: max-content !important;
                padding: 30px 20px !important;
                background: #ffffff !important;
                border-radius: 8px !important;
                border: 1px solid #e5e7eb !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
            }}
            
            /* Text đậm hơn */
            .displacy-word {{
                font-size: 16px !important;
                font-weight: 600 !important;
                color: #111827 !important;
                fill: #111827 !important;
            }}
            
            /* CRITICAL: Force arrow colors với !important */
            .displacy-arrow {{
                stroke-width: 3px !important;
            }}
            
            /* Positive arrows */
            .displacy-arrow[data-label="Positive"] {{
                stroke: #22c55e !important;
            }}
            
            /* Negative arrows */
            .displacy-arrow[data-label="Negative"] {{
                stroke: #ef4444 !important;
            }}
            
            /* Neutral arrows */
            .displacy-arrow[data-label="Neutral"] {{
                stroke: #6b7280 !important;
            }}
            
            /* Labels đậm và có màu */
            .displacy-label {{
                font-size: 13px !important;
                font-weight: 700 !important;
                fill: #ffffff !important;
                stroke: none !important;
            }}
            
            /* Label backgrounds */
            .displacy-label[data-label="Positive"] {{
                fill: #22c55e !important;
            }}
            
            .displacy-label[data-label="Negative"] {{
                fill: #ef4444 !important;
            }}
            
            .displacy-label[data-label="Neutral"] {{
                fill: #6b7280 !important;
            }}
            
            /* Arrow heads */
            marker path {{
                fill: currentColor !important;
            }}
        </style>
        {html}
    </div>
    """
    
    # Post-process HTML để inject data-label attributes
    for arc in arcs:
        polarity = arc['label']
        # Find và replace paths with data-label
        enhanced_html = enhanced_html.replace(
            f'<path class="displacy-arrow"',
            f'<path class="displacy-arrow" data-label="{polarity}"'
        )
        enhanced_html = enhanced_html.replace(
            f'<text class="displacy-label"',
            f'<text class="displacy-label" data-label="{polarity}"'
        )
    
    return enhanced_html