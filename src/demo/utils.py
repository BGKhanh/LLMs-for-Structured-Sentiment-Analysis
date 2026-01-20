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
    Chuyển SSA JSON thành HTML với entity highlights + dependency arcs.
    
    Args:
        text: Văn bản gốc.
        ssa_json: Dictionary kết quả SSA.
        
    Returns:
        HTML string với entities và dependencies.
    """
    nlp = get_spacy_model()
    doc = nlp(text)
    
    if isinstance(ssa_json, str):
        try:
            ssa_json = json.loads(ssa_json)
        except:
            return "<div>Invalid JSON</div>"

    opinions = ssa_json.get("opinions", [])
    if not opinions:
        return "<div style='padding: 20px;'>No opinions found.</div>"

    words = [token.text for token in doc]
    
    # Char to token mapping
    char_to_token = {}
    for token in doc:
        for i in range(token.idx, token.idx + len(token)):
            char_to_token[i] = token.i

    arcs = []
    entities = []  # NEW: For entity highlighting
    
    # Process opinions
    for idx, op in enumerate(opinions):
        try:
            # Extract components
            source_raw = op.get("Source", [[None]])[0][0]
            target_raw = op.get("Target", [[None]])[0][0]
            expr_raw = op.get("Polar_expression", [[None]])[0][0]
            polarity = op.get("Polarity", "Neutral")
            
            # Build entities for each component
            if source_raw:
                source_pos = extract_position(text, source_raw)
                source_span = parse_position_to_tuple(source_pos)
                if source_span and source_span != (0, 0):
                    entities.append({
                        "start": source_span[0],
                        "end": source_span[1],
                        "label": "Source"
                    })
            
            if target_raw:
                target_pos = extract_position(text, target_raw)
                target_span = parse_position_to_tuple(target_pos)
                if target_span and target_span != (0, 0):
                    entities.append({
                        "start": target_span[0],
                        "end": target_span[1],
                        "label": "Target"
                    })
            
            if expr_raw:
                expr_pos = extract_position(text, expr_raw)
                expr_span = parse_position_to_tuple(expr_pos)
                if expr_span and expr_span != (0, 0):
                    entities.append({
                        "start": expr_span[0],
                        "end": expr_span[1],
                        "label": "Expression"
                    })
                    
                    # Build arcs from Expression to Target
                    if target_raw:
                        target_pos_arc = extract_position(text, target_raw)
                        target_span_arc = parse_position_to_tuple(target_pos_arc)
                        
                        if expr_span and target_span_arc and expr_span != (0, 0) and target_span_arc != (0, 0):
                            start_tok = char_to_token.get(expr_span[0])
                            end_tok = char_to_token.get(target_span_arc[0])
                            
                            if start_tok is not None and end_tok is not None:
                                direction = "left" if start_tok > end_tok else "right"
                                arcs.append({
                                    "start": min(start_tok, end_tok),
                                    "end": max(start_tok, end_tok),
                                    "label": polarity,
                                    "dir": direction
                                })
            
        except Exception as e:
            print(f"⚠️ Error processing opinion {idx}: {e}")
            continue

    # Render entities first (bottom layer)
    ent_html = displacy.render(
        {"text": text, "ents": entities, "title": None},
        style="ent",
        manual=True,
        options={
            "colors": {
                "Source": "#ef4444",      # Red
                "Target": "#3b82f6",      # Blue
                "Expression": "#eab308"   # Yellow
            }
        },
        page=False
    )
    
    # Render dependencies (top layer)
    dep_data = {
        "words": [{"text": w, "tag": ""} for w in words],
        "arcs": arcs
    }
    
    dep_html = displacy.render(
        dep_data, 
        style="dep", 
        manual=True, 
        options={
            "compact": False,
            "distance": 140,
            "arrow_stroke": 4,
            "arrow_width": 12
        },
        page=False
    )
    
    # MERGE: Stack entities below dependencies
    combined_html = f"""
    <div style="width: 100%; overflow-x: auto; padding: 20px 0;">
        <style>
            /* Entity styles */
            mark.displacy-ent {{
                font-weight: 600 !important;
                padding: 3px 4px !important;
                border-radius: 3px !important;
                border-bottom: 3px solid !important;
            }}
            
            span.displacy-ent {{
                font-size: 13px !important;
                font-weight: 700 !important;
                padding: 2px 6px !important;
                border-radius: 3px !important;
                vertical-align: middle !important;
            }}
            
            /* Dependency styles */
            .displacy-arrow {{
                stroke-width: 4px !important;
                opacity: 1.0 !important;
            }}
            
            text.displacy-label {{
                font-size: 16px !important;
                font-weight: 800 !important;
                fill: #ffffff !important;
                paint-order: stroke fill !important;
                stroke-width: 18px !important;
                stroke-linecap: round !important;
            }}
            
            /* Polarity colors for dependency labels */
            .displacy-label-Positive {{
                stroke: #16a34a !important;
            }}
            
            .displacy-label-Negative {{
                stroke: #dc2626 !important;
            }}
            
            .displacy-label-Neutral {{
                stroke: #6b7280 !important;
            }}
        </style>
        
        <!-- Dependencies on top -->
        <div style="margin-bottom: -30px;">
            {dep_html}
        </div>
        
        <!-- Entities below -->
        <div style="margin-top: 10px;">
            {ent_html}
        </div>
    </div>
    """
    
    # Inject polarity classes to labels
    for arc in arcs:
        pol = arc['label']
        combined_html = combined_html.replace(
            f'>{pol}</text>',
            f' class="displacy-label-{pol}">{pol}</text>',
            1
        )
    
    return combined_html