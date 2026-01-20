# src/demo/utils.py

import json
import warnings
from src.utils.postprocessing import extract_position

warnings.filterwarnings("ignore")


def parse_position_to_tuple(position_str: str) -> tuple:
    """Convert position string to tuple."""
    try:
        parts = position_str.split(":")
        return (int(parts[0]), int(parts[1]))
    except:
        return None


def tokenize_text(text):
    """Simple tokenizer for Vietnamese text."""
    import re
    tokens = re.findall(r'\S+|\s+', text)
    
    result = []
    char_pos = 0
    for token in tokens:
        if token.strip():
            result.append({
                'text': token,
                'start': char_pos,
                'end': char_pos + len(token)
            })
        char_pos += len(token)
    
    return result


def convert_ssa_to_spacy(text, ssa_json):
    """
    Convert SSA JSON to custom HTML/SVG visualization.
    Fixed: High contrast text + Polarity annotations + Correct arrows.
    """
    if isinstance(ssa_json, str):
        try:
            ssa_json = json.loads(ssa_json)
        except:
            return "<div>Invalid JSON</div>"

    opinions = ssa_json.get("opinions", [])
    if not opinions:
        return "<div style='padding: 20px;'>No opinions found.</div>"

    tokens = tokenize_text(text)
    
    char_to_token = {}
    for idx, token in enumerate(tokens):
        for i in range(token['start'], token['end']):
            char_to_token[i] = idx

    entities_raw = []
    polarity_annotations = []  # NEW: Separate polarity annotations
    arcs = []
    
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
                        "text": source_raw,
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
                        "text": target_raw,
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
                        "text": expr_raw,
                        "opinion_idx": idx
                    })
                    
                    # FIX #2: Add Polarity annotation below Expression
                    polarity_annotations.append({
                        "start": expr_span[0],
                        "end": expr_span[1],
                        "polarity": polarity,
                        "opinion_idx": idx
                    })
                    
                    # FIX #3: Build CORRECT arc from Expression to Target
                    if target_raw:
                        target_pos_arc = extract_position(text, target_raw)
                        target_span_arc = parse_position_to_tuple(target_pos_arc)
                        
                        if target_span_arc and target_span_arc != (0, 0):
                            # Get token indices for BOTH start and end of spans
                            expr_start_tok = char_to_token.get(expr_span[0])
                            expr_end_tok = char_to_token.get(expr_span[1] - 1) if expr_span[1] > expr_span[0] else expr_start_tok
                            
                            target_start_tok = char_to_token.get(target_span_arc[0])
                            target_end_tok = char_to_token.get(target_span_arc[1] - 1) if target_span_arc[1] > target_span_arc[0] else target_start_tok
                            
                            if (expr_start_tok is not None and expr_end_tok is not None and 
                                target_start_tok is not None and target_end_tok is not None):
                                
                                # Use middle of each span for arc connection
                                expr_mid_tok = (expr_start_tok + expr_end_tok) // 2
                                target_mid_tok = (target_start_tok + target_end_tok) // 2
                                
                                arcs.append({
                                    "start_token": expr_mid_tok,
                                    "end_token": target_mid_tok,
                                    "label": polarity,
                                    "direction": "right" if expr_mid_tok < target_mid_tok else "left"
                                })
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
            continue

    # Deduplicate entities
    entity_groups = {}
    for ent in entities_raw:
        key = (ent['start'], ent['end'], ent['label'])
        if key not in entity_groups:
            entity_groups[key] = {
                "start": ent['start'],
                "end": ent['end'],
                "label": ent['label'],
                "text": ent['text'],
                "count": 0
            }
        entity_groups[key]['count'] += 1
    
    entities = list(entity_groups.values())
    
    # Generate HTML
    html = generate_visualization_html(text, tokens, entities, polarity_annotations, arcs)
    
    return html


def generate_visualization_html(text, tokens, entities, polarity_annotations, arcs):
    """Generate custom HTML/SVG visualization with high contrast."""
    
    # FIX #1: High contrast colors - ALL TEXT IS BLACK
    colors = {
        "Source": {"bg": "#fee2e2", "border": "#ef4444"},
        "Target": {"bg": "#dbeafe", "border": "#3b82f6"},
        "Expression": {"bg": "#fef3c7", "border": "#eab308"}
    }
    
    polarity_colors = {
        "Positive": "#16a34a",
        "Negative": "#dc2626",
        "Neutral": "#6b7280"
    }
    
    # Build polarity annotation map (expression position -> polarity)
    polarity_map = {}
    for pol_ann in polarity_annotations:
        key = (pol_ann['start'], pol_ann['end'])
        if key not in polarity_map:
            polarity_map[key] = []
        polarity_map[key].append(pol_ann['polarity'])
    
    # Build spans
    spans = []
    processed_indices = set()
    
    current_x = 50
    token_spacing = 15
    
    idx = 0
    while idx < len(tokens):
        if idx in processed_indices:
            idx += 1
            continue
        
        token = tokens[idx]
        
        # Check entity
        entity_info = None
        for ent in entities:
            if ent['start'] == token['start']:
                entity_info = ent
                break
        
        if entity_info:
            # Find tokens in span
            span_tokens = []
            span_text_parts = []
            
            for i in range(idx, len(tokens)):
                if tokens[i]['start'] >= entity_info['start'] and tokens[i]['end'] <= entity_info['end']:
                    span_tokens.append(tokens[i])
                    span_text_parts.append(tokens[i]['text'])
                    processed_indices.add(i)
                elif tokens[i]['start'] >= entity_info['end']:
                    break
            
            span_text = ' '.join(span_text_parts)
            label = entity_info['label']
            count_suffix = f" (×{entity_info['count']})" if entity_info['count'] > 1 else ""
            color = colors[label]
            
            span_width = len(span_text) * 10 + 50
            
            # FIX #2: Add Polarity annotation if this is Expression
            polarity_badge = ""
            if label == "Expression":
                span_key = (entity_info['start'], entity_info['end'])
                polarities = polarity_map.get(span_key, [])
                if polarities:
                    pol_text = ', '.join(polarities)
                    pol_color = polarity_colors.get(polarities[0], '#6b7280')
                    polarity_badge = f'''
                    <span style="
                        display: block;
                        font-size: 11px;
                        background: {pol_color};
                        color: white;
                        padding: 2px 6px;
                        border-radius: 3px;
                        margin-top: 2px;
                        font-weight: 700;
                        text-align: center;
                    ">Polarity: {pol_text}</span>
                    '''
            
            # FIX #1: Text màu đen hoàn toàn (#000000)
            span_html = f'''
                <span class="entity-span" style="
                    background: {color['bg']};
                    border-bottom: 3px solid {color['border']};
                    color: #000000;
                    padding: 4px 8px;
                    border-radius: 4px;
                    margin: 0 3px;
                    font-weight: 700;
                    display: inline-block;
                    font-size: 16px;
                ">
                    {span_text}
                    <span style="
                        font-size: 11px;
                        background: {color['border']};
                        color: white;
                        padding: 2px 6px;
                        border-radius: 3px;
                        margin-left: 4px;
                        font-weight: 700;
                    ">{label}{count_suffix}</span>
                    {polarity_badge}
                </span>
            '''
            
            spans.append({
                'html': span_html,
                'x': current_x + span_width / 2,
                'token_idx': idx,
                'width': span_width
            })
            
            current_x += span_width + token_spacing
            idx = max(processed_indices) + 1 if processed_indices else idx + 1
            
        else:
            # Regular token - FIX #1: Black text
            token_text = token['text']
            token_width = len(token_text) * 10 + 10
            
            token_html = f'<span class="token" style="margin: 0 2px; display: inline-block; color: #000000; font-size: 16px; font-weight: 500;">{token_text}</span>'
            
            spans.append({
                'html': token_html,
                'x': current_x + token_width / 2,
                'token_idx': idx,
                'width': token_width
            })
            
            current_x += token_width + token_spacing
            idx += 1
    
    # Build SVG arcs
    svg_width = current_x + 50
    svg_height = 150
    arc_y_base = 40
    
    token_to_span_x = {}
    for span in spans:
        token_to_span_x[span['token_idx']] = span['x']
    
    svg_arcs = []
    for arc_idx, arc in enumerate(arcs):
        start_idx = arc['start_token']
        end_idx = arc['end_token']
        
        # Find span positions
        start_x = token_to_span_x.get(start_idx)
        end_x = token_to_span_x.get(end_idx)
        
        # Nearest fallback
        if start_x is None:
            for i in range(start_idx, -1, -1):
                if i in token_to_span_x:
                    start_x = token_to_span_x[i]
                    break
        
        if end_x is None:
            for i in range(end_idx, len(tokens)):
                if i in token_to_span_x:
                    end_x = token_to_span_x[i]
                    break
        
        if start_x is None or end_x is None:
            continue
        
        # FIX #3: Correct arc path with proper direction
        mid_x = (start_x + end_x) / 2
        distance = abs(end_x - start_x)
        arc_height = min(distance / 2.5, 70)
        
        # Determine actual direction (Expression -> Target)
        if arc['direction'] == "right":
            # Expression is on the left, Target on the right
            path_start_x, path_end_x = start_x, end_x
        else:
            # Expression is on the right, Target on the left
            path_start_x, path_end_x = end_x, start_x
        
        # Quadratic bezier curve
        path_d = f"M {path_start_x},{arc_y_base} Q {mid_x},{arc_y_base - arc_height} {path_end_x},{arc_y_base}"
        
        arc_color = polarity_colors.get(arc['label'], '#6b7280')
        marker_id = f"arrow-{arc_idx}"
        
        svg_arcs.append(f'''
            <defs>
                <marker id="{marker_id}" markerWidth="12" markerHeight="12" 
                        refX="10" refY="3" orient="auto">
                    <path d="M0,0 L0,6 L9,3 z" fill="{arc_color}" />
                </marker>
            </defs>
            <path d="{path_d}" 
                  stroke="{arc_color}" 
                  stroke-width="4" 
                  fill="none" 
                  marker-end="url(#{marker_id})"
                  opacity="1.0"/>
            <text x="{mid_x}" y="{arc_y_base - arc_height - 8}" 
                  text-anchor="middle" 
                  font-size="16" 
                  font-weight="800"
                  fill="{arc_color}"
                  stroke="white"
                  stroke-width="4"
                  paint-order="stroke">
                {arc['label']}
            </text>
            <text x="{mid_x}" y="{arc_y_base - arc_height - 8}" 
                  text-anchor="middle" 
                  font-size="16" 
                  font-weight="800"
                  fill="{arc_color}">
                {arc['label']}
            </text>
        ''')
    
    # Combine HTML
    html = f'''
    <div style="width: 100%; background: white; padding: 30px 20px; border-radius: 8px; overflow-x: auto;">
        <div class="ssa-container">
            <!-- SVG Layer -->
            <svg width="{svg_width}" height="{svg_height}" style="position: absolute; top: 0; left: 0;">
                {''.join(svg_arcs)}
            </svg>
            
            <!-- Token/Entity Layer -->
            <div style="padding-top: {svg_height - 30}px; line-height: 3.2; position: relative;">
                {''.join([span['html'] for span in spans])}
            </div>
        </div>
        
        <!-- Legend -->
        <div style="margin-top: 30px; padding-top: 20px; border-top: 2px solid #e5e7eb;">
            <div style="display: flex; gap: 20px; flex-wrap: wrap; align-items: center;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: {colors['Source']['bg']}; border: 2px solid {colors['Source']['border']}; 
                                 padding: 4px 12px; border-radius: 4px; font-size: 13px; font-weight: 700; color: #000000;">Source</span>
                    <span style="color: #6b7280; font-size: 13px;">Nguồn</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: {colors['Target']['bg']}; border: 2px solid {colors['Target']['border']}; 
                                 padding: 4px 12px; border-radius: 4px; font-size: 13px; font-weight: 700; color: #000000;">Target</span>
                    <span style="color: #6b7280; font-size: 13px;">Đối tượng</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: {colors['Expression']['bg']}; border: 2px solid {colors['Expression']['border']}; 
                                 padding: 4px 12px; border-radius: 4px; font-size: 13px; font-weight: 700; color: #000000;">Expression</span>
                    <span style="color: #6b7280; font-size: 13px;">Biểu hiện</span>
                </div>
                <div style="margin-left: 20px; color: #6b7280; font-size: 13px;">
                    Arrows: <strong style="color: {polarity_colors['Positive']};">Positive</strong> | 
                    <strong style="color: {polarity_colors['Negative']};">Negative</strong> | 
                    <strong style="color: {polarity_colors['Neutral']};">Neutral</strong>
                </div>
            </div>
        </div>
    </div>
    '''
    
    return html