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
    """...(docstring giữ nguyên)..."""
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
    char_to_token = {}
    for token in doc:
        for i in range(token.idx, token.idx + len(token)):
            char_to_token[i] = token.i

    arcs = []
    entities_raw = []  # Collect all entities first
    
    # Process opinions
    for idx, op in enumerate(opinions):
        try:
            source_raw = op.get("Source", [[None]])[0][0]
            target_raw = op.get("Target", [[None]])[0][0]
            expr_raw = op.get("Polar_expression", [[None]])[0][0]
            polarity = op.get("Polarity", "Neutral")
            
            # Collect entities with opinion index
            if source_raw:
                source_pos = extract_position(text, source_raw)
                source_span = parse_position_to_tuple(source_pos)
                if source_span and source_span != (0, 0):
                    entities_raw.append({
                        "start": source_span[0],
                        "end": source_span[1],
                        "label": f"Source",
                        "opinion_idx": idx
                    })
            
            if target_raw:
                target_pos = extract_position(text, target_raw)
                target_span = parse_position_to_tuple(target_pos)
                if target_span and target_span != (0, 0):
                    entities_raw.append({
                        "start": target_span[0],
                        "end": target_span[1],
                        "label": f"Target",
                        "opinion_idx": idx
                    })
            
            if expr_raw:
                expr_pos = extract_position(text, expr_raw)
                expr_span = parse_position_to_tuple(expr_pos)
                if expr_span and expr_span != (0, 0):
                    entities_raw.append({
                        "start": expr_span[0],
                        "end": expr_span[1],
                        "label": f"Expression",
                        "opinion_idx": idx
                    })
                    
                    # Build arcs
                    if target_raw:
                        target_pos_arc = extract_position(text, target_raw)
                        target_span_arc = parse_position_to_tuple(target_pos_arc)
                        
                        if expr_span and target_span_arc:
                            start_tok = char_to_token.get(expr_span[0])
                            end_tok = char_to_token.get(target_span_arc[0])
                            
                            if start_tok is not None and end_tok is not None:
                                arcs.append({
                                    "start": min(start_tok, end_tok),
                                    "end": max(start_tok, end_tok),
                                    "label": polarity,
                                    "dir": "left" if start_tok > end_tok else "right"
                                })
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue

    # === DEDUPLICATE ENTITIES ===
    # Group by (start, end, label) and count occurrences
    entity_groups = {}
    for ent in entities_raw:
        key = (ent['start'], ent['end'], ent['label'])
        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(ent['opinion_idx'])
    
    # Build deduplicated entities with count suffix if needed
    entities = []
    for (start, end, label), opinion_indices in entity_groups.items():
        count = len(opinion_indices)
        if count > 1:
            # Add count suffix
            final_label = f"{label} (×{count})"
        else:
            final_label = label
        
        entities.append({
            "start": start,
            "end": end,
            "label": final_label
        })
    
    # Update color mapping to handle count suffixes
    color_mapping = {
        "Source": "#ef4444",
        "Target": "#3b82f6",
        "Expression": "#eab308"
    }
    
    # Build colors dict with all possible labels (including counts)
    colors = {}
    for label in ["Source", "Target", "Expression"]:
        colors[label] = color_mapping[label]
        for i in range(2, 10):  # Support up to ×9
            colors[f"{label} (×{i})"] = color_mapping[label]
    
    # Render entities
    ent_html = displacy.render(
        {"text": text, "ents": entities, "title": None},
        style="ent",
        manual=True,
        options={"colors": colors},
        page=False
    )
    
    # Render dependencies
    dep_html = displacy.render(
        {"words": [{"text": w, "tag": ""} for w in words], "arcs": arcs},
        style="dep",
        manual=True,
        options={"compact": False, "distance": 140, "arrow_stroke": 4, "arrow_width": 12},
        page=False
    )
    
    # Combine with ENHANCED CSS
    combined_html = f"""
    <div style="width: 100%; overflow-x: auto; padding: 20px 0;">
        <style>
            /* Entity highlights */
            mark.displacy-ent {{
                font-weight: 700 !important;
                padding: 4px 6px !important;
                border-radius: 4px !important;
                border-bottom: 4px solid !important;
                font-size: 16px !important;
            }}
            
            span.displacy-ent {{
                font-size: 14px !important;
                font-weight: 800 !important;
                padding: 3px 8px !important;
                border-radius: 4px !important;
                margin-left: 4px !important;
            }}
            
            /* Dependency arrows - RÕ RÀNG */
            .displacy-arrow {{
                stroke-width: 5px !important;
                opacity: 1.0 !important;
            }}
            
            /* Arrow labels - RÕ VÀ ĐẬM */
            text.displacy-label {{
                font-size: 18px !important;
                font-weight: 900 !important;
                fill: #ffffff !important;
                paint-order: stroke fill !important;
                stroke-width: 22px !important;
                stroke-linecap: round !important;
            }}
            
            /* Polarity-specific colors for arcs */
            path[data-arc*="Positive"] {{
                stroke: #16a34a !important;
            }}
            
            path[data-arc*="Negative"] {{
                stroke: #dc2626 !important;
            }}
            
            path[data-arc*="Neutral"] {{
                stroke: #6b7280 !important;
            }}
        </style>
        
        <div style="margin-bottom: -20px;">
            {dep_html}
        </div>
        
        <div style="margin-top: 15px;">
            {ent_html}
        </div>
    </div>
    """
    
    # Inject data attributes for styling
    for arc in arcs:
        pol = arc['label']
        # Add data-arc attribute to path elements
        combined_html = combined_html.replace(
            '<path class="displacy-arrow"',
            f'<path class="displacy-arrow" data-arc="{pol}"',
            1
        )
        # Add class to label text
        combined_html = combined_html.replace(
            f'class="displacy-label">{pol}</text>',
            f'class="displacy-label" style="stroke: {"#16a34a" if pol=="Positive" else "#dc2626" if pol=="Negative" else "#6b7280"} !important;">{pol}</text>',
            1
        )
    
    return combined_html