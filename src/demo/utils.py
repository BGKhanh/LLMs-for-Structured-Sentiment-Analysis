# src/demo/utils.py

import json
import warnings
from src.utils.postprocessing import extract_position

warnings.filterwarnings("ignore")


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


def tokenize_text(text):
    """Simple tokenizer for Vietnamese text."""
    import re
    # Split by spaces and punctuation
    tokens = re.findall(r'\S+|\s+', text)
    
    result = []
    char_pos = 0
    for token in tokens:
        if token.strip():  # Non-whitespace token
            result.append({
                'text': token,
                'start': char_pos,
                'end': char_pos + len(token)
            })
        char_pos += len(token)
    
    return result


def convert_ssa_to_spacy(text, ssa_json):
    """
    Convert SSA JSON output to custom HTML/SVG visualization.
    No spaCy dependency - pure HTML/CSS/SVG implementation.
    """
    if isinstance(ssa_json, str):
        try:
            ssa_json = json.loads(ssa_json)
        except:
            return "<div>Invalid JSON</div>"

    opinions = ssa_json.get("opinions", [])
    if not opinions:
        return "<div style='padding: 20px;'>No opinions found.</div>"

    # Tokenize text
    tokens = tokenize_text(text)
    
    # Build char-to-token mapping
    char_to_token = {}
    for idx, token in enumerate(tokens):
        for i in range(token['start'], token['end']):
            char_to_token[i] = idx

    # Collect entities and arcs
    entities_raw = []
    arcs = []
    
    for idx, op in enumerate(opinions):
        try:
            source_raw = op.get("Source", [[None]])[0][0]
            target_raw = op.get("Target", [[None]])[0][0]
            expr_raw = op.get("Polar_expression", [[None]])[0][0]
            polarity = op.get("Polarity", "Neutral")
            
            source_span = None
            target_span = None
            expr_span = None
            
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
                        "opinion_idx": idx,
                        "polarity": None
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
                        "opinion_idx": idx,
                        "polarity": None
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
                        "opinion_idx": idx,
                        "polarity": polarity  # Store polarity with Expression
                    })
            
            # Build arcs: Source -> Expression -> Target
            if source_raw and expr_raw and source_span and expr_span:
                source_tok = char_to_token.get(source_span[0])
                expr_tok = char_to_token.get(expr_span[0])
                
                if source_tok is not None and expr_tok is not None:
                    arcs.append({
                        "start_token": min(source_tok, expr_tok),
                        "end_token": max(source_tok, expr_tok),
                        "label": polarity,
                        "direction": "right" if source_tok < expr_tok else "left",
                        "arc_type": "source_to_expr"
                    })
            
            if expr_raw and target_raw and expr_span and target_span:
                expr_tok = char_to_token.get(expr_span[0])
                target_tok = char_to_token.get(target_span[0])
                
                if expr_tok is not None and target_tok is not None:
                    arcs.append({
                        "start_token": min(expr_tok, target_tok),
                        "end_token": max(expr_tok, target_tok),
                        "label": polarity,
                        "direction": "right" if expr_tok < target_tok else "left",
                        "arc_type": "expr_to_target"
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
                "count": 0,
                "polarity": ent.get('polarity')  # Keep polarity info
            }
        entity_groups[key]['count'] += 1
    
    entities = list(entity_groups.values())
    
    # Generate HTML
    html = generate_visualization_html(text, tokens, entities, arcs)
    
    return html


def generate_visualization_html(text, tokens, entities, arcs):
    """Generate custom HTML/SVG visualization."""
    
    # Colors
    colors = {
        "Source": {"bg": "#fee2e2", "border": "#ef4444", "text": "#000000"},
        "Target": {"bg": "#dbeafe", "border": "#3b82f6", "text": "#000000"},
        "Expression": {"bg": "#fef3c7", "border": "#eab308", "text": "#000000"}
    }
    
    polarity_colors = {
        "Positive": "#16a34a",
        "Negative": "#dc2626",
        "Neutral": "#6b7280"
    }
    
    # Build spans (group tokens by entity)
    spans = []
    processed_indices = set()
    
    current_x = 50  # Starting X position
    token_spacing = 15
    
    idx = 0
    while idx < len(tokens):
        if idx in processed_indices:
            idx += 1
            continue
        
        token = tokens[idx]
        
        # Check if this token starts an entity span
        entity_info = None
        for ent in entities:
            if ent['start'] == token['start']:
                entity_info = ent
                break
        
        if entity_info:
            # Find all tokens in this entity span
            span_tokens = []
            span_text_parts = []
            
            for i in range(idx, len(tokens)):
                if tokens[i]['start'] >= entity_info['start'] and tokens[i]['end'] <= entity_info['end']:
                    span_tokens.append(tokens[i])
                    span_text_parts.append(tokens[i]['text'])
                    processed_indices.add(i)
                elif tokens[i]['start'] >= entity_info['end']:
                    break
            
            # Create span HTML
            span_text = ' '.join(span_text_parts)
            label = entity_info['label']
            count_suffix = f" (×{entity_info['count']})" if entity_info['count'] > 1 else ""
            color = colors[label]
            polarity = entity_info.get('polarity')
            
            span_width = len(span_text) * 10 + 40
            
            # If Expression with Polarity, add polarity label below
            if label == "Expression" and polarity:
                polarity_color = polarity_colors.get(polarity, '#6b7280')
                span_html = f'''
                    <span class="entity-span" style="
                        display: inline-flex;
                        flex-direction: column;
                        align-items: center;
                        margin: 0 3px;
                        vertical-align: top;
                    ">
                        <span style="
                            background: {color['bg']};
                            border-bottom: 3px solid {color['border']};
                            color: {color['text']};
                            padding: 4px 8px;
                            border-radius: 4px;
                            font-weight: 600;
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
                                white-space: nowrap;
                            ">{label}{count_suffix}</span>
                        </span>
                        <span style="
                            background: {polarity_color};
                            color: white;
                            padding: 3px 10px;
                            border-radius: 4px;
                            font-size: 11px;
                            font-weight: 700;
                            margin-top: 4px;
                            white-space: nowrap;
                        ">Polarity: {polarity}</span>
                    </span>
                '''
            else:
                # Regular entity (Source or Target)
                span_html = f'''
                    <span class="entity-span" style="
                        background: {color['bg']};
                        border-bottom: 3px solid {color['border']};
                        color: {color['text']};
                        padding: 4px 8px;
                        border-radius: 4px;
                        margin: 0 3px;
                        font-weight: 600;
                        display: inline-block;
                        vertical-align: top;
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
                            white-space: nowrap;
                        ">{label}{count_suffix}</span>
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
            # Regular token (not part of any entity)
            token_text = token['text']
            token_width = len(token_text) * 10 + 10
            
            token_html = f'<span class="token" style="margin: 0 2px; display: inline-block; color: #000000;">{token_text}</span>'
            
            spans.append({
                'html': token_html,
                'x': current_x + token_width / 2,
                'token_idx': idx,
                'width': token_width
            })
            
            current_x += token_width + token_spacing
            idx += 1
    
    # Build SVG for arcs using span positions
    svg_width = current_x + 50
    svg_height = 150
    arc_y_base = 40
    
    # Map token indices to span positions
    token_to_span_x = {}
    for span in spans:
        token_to_span_x[span['token_idx']] = span['x']
    
    svg_arcs = []
    for arc in arcs:
        start_idx = arc['start_token']
        end_idx = arc['end_token']
        
        # Find the closest span positions
        start_x = token_to_span_x.get(start_idx)
        end_x = token_to_span_x.get(end_idx)
        
        # If exact match not found, find nearest
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
        
        # Calculate arc path
        mid_x = (start_x + end_x) / 2
        distance = abs(end_x - start_x)
        arc_height = min(distance / 3, 60)
        
        # SVG path for curved arc
        path_d = f"M {start_x},{arc_y_base} Q {mid_x},{arc_y_base - arc_height} {end_x},{arc_y_base}"
        
        # Arc color based on polarity
        arc_color = polarity_colors.get(arc['label'], '#6b7280')
        
        # Arrow marker
        marker_id = f"arrow-{len(svg_arcs)}"
        
        svg_arcs.append(f'''
            <defs>
                <marker id="{marker_id}" markerWidth="10" markerHeight="10" 
                        refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                    <path d="M0,0 L0,6 L9,3 z" fill="{arc_color}" />
                </marker>
            </defs>
            <path d="{path_d}" 
                  stroke="{arc_color}" 
                  stroke-width="3" 
                  fill="none" 
                  marker-end="url(#{marker_id})"
                  opacity="0.9"/>
            <text x="{mid_x}" y="{arc_y_base - arc_height - 5}" 
                  text-anchor="middle" 
                  font-size="14" 
                  font-weight="700"
                  fill="{arc_color}"
                  stroke="white"
                  stroke-width="3"
                  paint-order="stroke">
                {arc['label']}
            </text>
            <text x="{mid_x}" y="{arc_y_base - arc_height - 5}" 
                  text-anchor="middle" 
                  font-size="14" 
                  font-weight="700"
                  fill="{arc_color}">
                {arc['label']}
            </text>
        ''')
    
    # Combine everything
    html = f'''
    <div style="width: 100%; background: white; padding: 30px 20px; border-radius: 8px; overflow-x: auto;">
        <style>
            .ssa-container {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                position: relative;
            }}
            .token {{
                font-size: 16px;
                display: inline-block;
                color: #000000;
            }}
        </style>
        
        <div class="ssa-container">
            <!-- SVG Layer for Arcs -->
            <svg width="{svg_width}" height="{svg_height}" style="position: absolute; top: 0; left: 0; pointer-events: none;">
                {''.join(svg_arcs)}
            </svg>
            
            <!-- Token Layer -->
            <div style="padding-top: {svg_height - 40}px; line-height: 2.8; position: relative;">
                {''.join([span['html'] for span in spans])}
            </div>
        </div>
        
        <!-- Legend -->
        <div style="margin-top: 30px; padding-top: 20px; border-top: 2px solid #e5e7eb;">
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: {colors['Source']['bg']}; border: 2px solid {colors['Source']['border']}; 
                                 padding: 4px 12px; border-radius: 4px; font-size: 13px; font-weight: 600; color: #000000;">Source</span>
                    <span style="color: #000000; font-size: 13px;">Nguồn cảm xúc</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: {colors['Target']['bg']}; border: 2px solid {colors['Target']['border']}; 
                                 padding: 4px 12px; border-radius: 4px; font-size: 13px; font-weight: 600; color: #000000;">Target</span>
                    <span style="color: #000000; font-size: 13px;">Đối tượng</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: {colors['Expression']['bg']}; border: 2px solid {colors['Expression']['border']}; 
                                 padding: 4px 12px; border-radius: 4px; font-size: 13px; font-weight: 600; color: #000000;">Expression</span>
                    <span style="color: #000000; font-size: 13px;">Biểu hiện cảm xúc</span>
                </div>
            </div>
        </div>
    </div>
    '''
    
    return html