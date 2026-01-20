# src/demo/utils.py

import spacy
from spacy import displacy
import json
import warnings
from src.utils.postprocessing import extract_position
import re

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
    """
    try:
        parts = position_str.split(":")
        return (int(parts[0]), int(parts[1]))
    except:
        return None


def convert_ssa_to_spacy(text, ssa_json):
    """
    Convert SSA JSON output to spaCy displacy visualization.
    Fixed version with SVG manipulation for proper arc visibility.
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
    char_to_token = {}
    for token in doc:
        for i in range(token.idx, token.idx + len(token)):
            char_to_token[i] = token.i

    arcs = []
    entities_raw = []
    
    # Process opinions
    for idx, op in enumerate(opinions):
        try:
            source_raw = op.get("Source", [[None]])[0][0]
            target_raw = op.get("Target", [[None]])[0][0]
            expr_raw = op.get("Polar_expression", [[None]])[0][0]
            polarity = op.get("Polarity", "Neutral")
            
            # Collect entities
            if source_raw:
                source_pos = extract_position(text, source_raw)
                source_span = parse_position_to_tuple(source_pos)
                if source_span and source_span != (0, 0):
                    entities_raw.append({
                        "start": source_span[0],
                        "end": source_span[1],
                        "label": "Source",
                        "opinion_idx": idx
                    })
            
            if target_raw:
                target_pos = extract_position(text, target_raw)
                target_span = parse_position_to_tuple(target_pos)
                if target_span and target_span != (0, 0):
                    entities_raw.append({
                        "start": target_span[0],
                        "end": target_span[1],
                        "label": "Target",
                        "opinion_idx": idx
                    })
            
            if expr_raw:
                expr_pos = extract_position(text, expr_raw)
                expr_span = parse_position_to_tuple(expr_pos)
                if expr_span and expr_span != (0, 0):
                    entities_raw.append({
                        "start": expr_span[0],
                        "end": expr_span[1],
                        "label": "Expression",
                        "opinion_idx": idx
                    })
                    
                    # Build arcs from Expression to Target
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
            print(f"⚠️ Error processing opinion {idx}: {e}")
            continue

    # Deduplicate entities
    entity_groups = {}
    for ent in entities_raw:
        key = (ent['start'], ent['end'], ent['label'])
        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(ent['opinion_idx'])
    
    entities = []
    for (start, end, label), opinion_indices in entity_groups.items():
        count = len(opinion_indices)
        final_label = f"{label} (×{count})" if count > 1 else label
        entities.append({
            "start": start,
            "end": end,
            "label": final_label
        })
    
    # Color mapping
    color_mapping = {
        "Source": "#ef4444",
        "Target": "#3b82f6",
        "Expression": "#eab308"
    }
    
    colors = {}
    for label in ["Source", "Target", "Expression"]:
        colors[label] = color_mapping[label]
        for i in range(2, 10):
            colors[f"{label} (×{i})"] = color_mapping[label]
    
    # Render dependency graph (arcs) with custom options
    dep_html = displacy.render(
        {"words": [{"text": w, "tag": ""} for w in words], "arcs": arcs},
        style="dep",
        manual=True,
        options={
            "compact": False, 
            "distance": 120,
            "arrow_stroke": 3,
            "arrow_width": 12,
            "word_spacing": 25,
            "bg": "#ffffff"
        },
        page=False
    )
    
    # Render entities
    ent_html = displacy.render(
        {"text": text, "ents": entities, "title": None},
        style="ent",
        manual=True,
        options={"colors": colors},
        page=False
    )
    
    # CRITICAL: Modify SVG to increase arc stroke and opacity
    # Find and replace arc paths in dep_html
    for i, arc in enumerate(arcs):
        pol = arc['label']
        
        # Determine color based on polarity
        if pol.lower() == 'positive':
            arc_color = '#16a34a'
        elif pol.lower() == 'negative':
            arc_color = '#dc2626'
        else:
            arc_color = '#6b7280'
        
        # Replace the arc path with enhanced styling
        dep_html = re.sub(
            r'<path class="displacy-arrow"[^>]*fill="none"[^>]*/>',
            f'<path class="displacy-arrow" fill="none" stroke="{arc_color}" stroke-width="4" opacity="1.0"/>',
            dep_html,
            count=1
        )
    
    # Enhance label text styling in SVG
    dep_html = re.sub(
        r'<text class="displacy-label"([^>]*)>',
        r'<text class="displacy-label"\1 style="font-weight: 900; font-size: 16px;">',
        dep_html
    )
    
    # Combine with proper CSS
    combined_html = f"""
    <div style="width: 100%; overflow-x: auto; padding: 20px 0; background: #ffffff;">
        <style>
            /* Force SVG elements to be visible */
            svg.displacy {{
                background: transparent !important;
            }}
            
            /* Make arrows BOLD and VISIBLE */
            .displacy-arrow {{
                stroke-width: 4px !important;
                opacity: 1.0 !important;
                fill: none !important;
            }}
            
            /* Arc labels - white background with dark text */
            text.displacy-label {{
                font-size: 16px !important;
                font-weight: 900 !important;
                fill: #000000 !important;
                paint-order: stroke fill !important;
                stroke: #ffffff !important;
                stroke-width: 6px !important;
                stroke-linecap: round !important;
                stroke-linejoin: round !important;
            }}
            
            /* Word spacing */
            text.displacy-token {{
                font-size: 16px !important;
            }}
            
            /* Entity highlights */
            mark.displacy-ent {{
                font-weight: 700 !important;
                padding: 4px 6px !important;
                border-radius: 4px !important;
                border-bottom: 3px solid !important;
                font-size: 16px !important;
                line-height: 2.5 !important;
                display: inline-block !important;
            }}
            
            span.displacy-ent {{
                font-size: 13px !important;
                font-weight: 800 !important;
                padding: 3px 7px !important;
                border-radius: 4px !important;
                margin-left: 4px !important;
                vertical-align: middle !important;
                display: inline-block !important;
            }}
            
            /* Container spacing */
            .ssa-viz {{
                background: white;
                padding: 20px;
                border-radius: 8px;
            }}
            
            .dep-container {{
                margin-bottom: 10px;
                min-height: 120px;
            }}
            
            .ent-container {{
                margin-top: 10px;
                padding-top: 10px;
            }}
        </style>
        
        <div class="ssa-viz">
            <!-- Dependency arcs -->
            <div class="dep-container">
                {dep_html}
            </div>
            
            <!-- Entity labels -->
            <div class="ent-container">
                {ent_html}
            </div>
        </div>
    </div>
    """
    
    return combined_html