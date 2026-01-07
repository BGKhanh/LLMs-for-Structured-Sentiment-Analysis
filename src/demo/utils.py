import spacy
from spacy import displacy
import json
import warnings

# Tắt warnings của spaCy nếu model ngôn ngữ không khớp hoàn toàn
warnings.filterwarnings("ignore")

def get_spacy_model():
    """Load blank spaCy model for visualization container."""
    # Dùng blank model để không cần download language pack nặng
    # Chỉ dùng để chứa text và render tokens
    try:
        nlp = spacy.blank("vi")
    except:
        nlp = spacy.blank("en")
    return nlp

def find_span_indices(text, substring):
    """
    Tìm vị trí start/end char của substring trong text.
    Trả về (start_char, end_char) của lần xuất hiện đầu tiên.
    """
    if not substring or not isinstance(substring, str):
        return None
    
    start = text.find(substring)
    if start == -1:
        return None
    return start, start + len(substring)

def convert_ssa_to_spacy(text, ssa_json):
    """
    Chuyển đổi kết quả SSA JSON thành HTML Dependency Graph.
    
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

    # Định nghĩa màu sắc cho các Polarity
    colors = {
        "Positive": "#28a745", # Green
        "Negative": "#dc3545", # Red
        "Neutral": "#6c757d",  # Gray
        "expression": "#007bff", # Blue highlight
        "target": "#fd7e14"      # Orange highlight
    }
    
    if isinstance(ssa_json, str):
        try:
            ssa_json = json.loads(ssa_json)
        except:
            return "<div>Invalid JSON Format</div>"

    opinions = ssa_json.get("opinions", [])
    if not opinions:
        return f"<div style='padding: 20px;'>No opinions found in text.</div>"

    # Xây dựng cấu trúc cho displacy manual render
    # displaCy manual cần: {text: "...", ents: [], title: None} cho entity
    # Nhưng để vẽ ARROWS (Dep parsing), ta cần trick một chút bằng cách gán .dep_ cho tokens
    
    # Tuy nhiên, displacy.render(doc, style="dep") yêu cầu Doc object chuẩn.
    # Cách tốt nhất để vẽ custom arrows là manual render với 'arcs'.
    
    words = [token.text for token in doc]
    
    # Tạo mapping từ char index sang token index
    char_to_token = {}
    for token in doc:
        for i in range(token.idx, token.idx + len(token)):
            char_to_token[i] = token.i

    arcs = []
    
    # Duyệt qua các opinions để tạo arrows
    for op in opinions:
        # Lấy text từ mảng [["text"], ["span"]]
        # Giả sử phần tử đầu tiên là text string
        try:
            # Handle format variants (list of lists vs simple string)
            expr_raw = op.get("Polar_expression", [[None]])[0][0]
            target_raw = op.get("Target", [[None]])[0][0]
            polarity = op.get("Polarity", "Neutral")
            
            if not expr_raw or not target_raw:
                continue

            # Tìm vị trí trong text
            expr_span = find_span_indices(text, expr_raw)
            target_span = find_span_indices(text, target_raw)
            
            if expr_span and target_span:
                # Map sang token index
                # Lấy token đầu tiên của span làm anchor
                start_token_idx = char_to_token.get(expr_span[0])
                end_token_idx = char_to_token.get(target_span[0])
                
                if start_token_idx is not None and end_token_idx is not None:
                    # Tạo arrow từ Expression -> Target
                    # dir=1: left to right, -1: right to left
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
    
    # Inject CSS colors for specific polarity labels if needed (optional)
    # displacy dùng class .displacy-label
    
    return html