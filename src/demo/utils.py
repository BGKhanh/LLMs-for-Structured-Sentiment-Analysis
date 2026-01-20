# src/demo/utils.py

import spacy
from spacy import displacy
import json
import warnings
from src.utils.postprocessing import extract_position

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
    Fixed version with proper layering: arcs on top, entities below.
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
    
    # Render dependency graph (arcs)
    dep_html = displacy.render(
        {"words": [{"text": w, "tag": ""} for w in words], "arcs": arcs},
        style="dep",
        manual=True,
        options={
            "compact": False, 
            "distance": 150,
            "arrow_stroke": 2.5,
            "arrow_width": 10,
            "offset_x": 0
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
    
    # CRITICAL FIX: Reverse order - dependency ABOVE, entities BELOW
    combined_html = f"""
    <div style="width: 100%; overflow-x: auto; padding: 20px 0; background: white;">
        <style>
            /* Container for proper layering */
            .ssa-container {{
                position: relative;
                background: white;
            }}
            
            /* Dependency layer - ABOVE */
            .dep-layer {{
                position: relative;
                z-index: 10;
                margin-bottom: -30px; /* Overlap with entity layer */
                padding-bottom: 40px;
            }}
            
            /* Entity layer - BELOW */
            .ent-layer {{
                position: relative;
                z-index: 1;
                padding-top: 10px;
                background: white;
            }}
            
            /* Dependency arrows styling */
            .displacy-arrow {{
                stroke-width: 2.5px !important;
                fill: none !important;
            }}
            
            /* Arc labels - positioned ABOVE the curve */
            text.displacy-label {{
                font-size: 14px !important;
                font-weight: 700 !important;
                fill: #1f2937 !important;
                paint-order: stroke fill !important;
                stroke: #ffffff !important;
                stroke-width: 4px !important;
                stroke-linecap: round !important;
                stroke-linejoin: round !important;
            }}
            
            /* Entity highlights */
            mark.displacy-ent {{
                font-weight: 600 !important;
                padding: 3px 5px !important;
                border-radius: 3px !important;
                border-bottom: 3px solid !important;
                font-size: 15px !important;
                line-height: 2.2 !important;
            }}
            
            span.displacy-ent {{
                font-size: 12px !important;
                font-weight: 700 !important;
                padding: 2px 6px !important;
                border-radius: 3px !important;
                margin-left: 3px !important;
                vertical-align: middle !important;
            }}
            
            /* Polarity-specific colors */
            .arc-positive {{
                stroke: #16a34a !important;
            }}
            
            .arc-negative {{
                stroke: #dc2626 !important;
            }}
            
            .arc-neutral {{
                stroke: #6b7280 !important;
            }}
        </style>
        
        <div class="ssa-container">
            <!-- Dependency layer FIRST (on top) -->
            <div class="dep-layer">
                {dep_html}
            </div>
            
            <!-- Entity layer SECOND (below) -->
            <div class="ent-layer">
                {ent_html}
            </div>
        </div>
    </div>
    """
    
    # Apply polarity colors to arcs
    for arc in arcs:
        pol = arc['label'].lower()
        arc_class = f"arc-{pol}"
        combined_html = combined_html.replace(
            '<path class="displacy-arrow"',
            f'<path class="displacy-arrow {arc_class}"',
            1
        )
    
    return combined_html