"""
SemEval Vietnamese Structured Sentiment Analysis Task Utilities
for LM Evaluation Harness

This module provides helper functions for evaluating LLMs on Vietnamese
structured sentiment analysis using the SemEval-2022 Task 10 format.
"""

import json
import re
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path

import datasets

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter, register_aggregation

# Setup logging
eval_logger = logging.getLogger(__name__)

# Import SemEval evaluation utilities
import sys
semeval_path = Path(__file__).parent.parent.parent.parent / "semeval22_structured_sentiment" / "evaluation"
sys.path.insert(0, str(semeval_path))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "utils"))
from postprocessing import extract_position 

try:
    from evaluate import convert_opinion_to_tuple, tuple_f1, tuple_precision, tuple_recall
    SEMEVAL_AVAILABLE = True
except ImportError as e:
    eval_logger.warning(f"SemEval evaluation module not available: {e}")
    SEMEVAL_AVAILABLE = False


# ==================== DATA PROCESSING ====================

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Validate and normalize SemEval format dataset.
    
    Args:
        dataset: HuggingFace Dataset with fields [sent_id, text, opinions]
        
    Returns:
        Validated dataset
        
    Raises:
        AssertionError: If required fields are missing
    """
    def _validate_doc(doc: Dict) -> Dict:
        """Validate single document structure."""
        # Check required fields
        assert "sent_id" in doc, "Missing 'sent_id' field"
        assert "text" in doc, "Missing 'text' field"
        assert "opinions" in doc, "Missing 'opinions' field"
        
        # Validate opinions structure
        if not isinstance(doc["opinions"], list):
            eval_logger.warning(f"sent_id {doc['sent_id']}: opinions is not a list")
            doc["opinions"] = []
        
        # Ensure sent_id is int
        if isinstance(doc["sent_id"], str):
            doc["sent_id"] = int(doc["sent_id"])
        
        return doc
    
    return dataset.map(_validate_doc)


def load_fewshot_cot_examples() -> List[Dict]:
    """
    Load hardcoded Chain-of-Thought examples with reasoning.
    
    These examples are curated manually and include step-by-step reasoning
    demonstrations to guide the model's thinking process.
    
    Returns:
        List of example dicts with 'text', 'reasoning', and 'output' fields
    """
    examples_path = Path(__file__).parent / "examples_cot.json"
    
    if not examples_path.exists():
        eval_logger.warning(f"CoT examples file not found: {examples_path}")
        return []
    
    with open(examples_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    eval_logger.info(f"Loaded {len(examples)} CoT examples from {examples_path}")
    return examples


# ==================== JSON EXTRACTION ====================

@register_filter("extract_json")
class ExtractJSONFilter(Filter):
    """
    Filter to extract JSON from model responses.
    
    This filter handles various output formats from different models:
    - Pure JSON
    - JSON in markdown code blocks
    - JSON mixed with text/reasoning
    
    Registered with the framework for use in filter_list configs.
    """
    
    def __init__(self, **kwargs) -> None:
        """Initialize filter."""
        super().__init__(**kwargs)
    
    def apply(
        self, 
        resps: List[List[str]], 
        docs: List[Dict]
    ) -> List[List[str]]:
        """
        Apply JSON extraction to response lists.
        
        Args:
            resps: List of response lists (per document)
            docs: List of document dicts
            
        Returns:
            List of filtered response lists (JSON strings)
        """
        filtered_resps = []
        
        for resp_list in resps:
            filtered_resp_list = []
            for resp in resp_list:
                extracted = extract_json_from_response(resp)
                filtered_resp_list.append(extracted)
            filtered_resps.append(filtered_resp_list)
        
        return filtered_resps


def extract_json_from_response(response: str) -> str:
    """
    Universal JSON extractor for all model types.
    
    Handles multiple output patterns:
    1. Pure JSON: {"sent_id": ...}
    2. Markdown code blocks:\\n{...}\\n    3. Text prefix: "Here is the analysis:\\n{...}"
    4. Reasoning + JSON: "Step 1: ...\\nResult: {...}"
    
    Args:
        response: Raw model output string
        
    Returns:
        Extracted JSON string (not parsed) or "{}" if extraction fails
        
    Examples:
        >>> extract_json_from_response('{"sent_id": 1}')
        '{"sent_id": 1}'
        
        >>> extract_json_from_response('\\n{"sent_id": 1}\\n```')
        '{"sent_id": 1}'
    """
    if not response or not response.strip():
        return '{"opinions": []}'
    
    response = response.strip()
    
    think_block_pattern = r"<think>[\s\S]*?</think>\s*"
    response = re.sub(think_block_pattern, "", response, flags=re.DOTALL)
    
    # Handle <|thought|> marker (alternative format)
    if "<|thought|>" in response:
        parts = response.split("<|thought|>")
        response = parts[-1].strip()
    
    assistant_markers = ["assistant:", "assistant", "<assistant>"]
    for marker in assistant_markers:
        if marker in response:
            # Split and take content after marker
            parts = response.split(marker, 1)
            if len(parts) > 1:
                response = parts[1].strip()
                break
            
    # Pattern 1: Markdown code block
    markdown_patterns = r'\s*([\s\S]*?)\s*```'
    
    match = re.search(markdown_patterns, response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        if json_str.startswith('{'):
            # Validate before returning
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                continue
    
    # Pattern 2: Find largest JSON object with proper nesting
    # This handles {"opinions": [...]} even with nested braces
    try:
        brace_depth = 0
        start_idx = None
        best_json = None

        for i, char in enumerate(response):
            if char == '{':
                if brace_depth == 0:
                    start_idx = i
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
                if brace_depth == 0 and start_idx is not None:
                    # Found a complete JSON object
                    json_candidate = response[start_idx:i+1]
                    # Quick validation: try to parse
                    try:
                        json.loads(json_candidate)
                        if best_json is None or len(json_candidate) > len(best_json):
                            best_json = json_candidate
                    except json.JSONDecodeError:
                        pass
                    start_idx = None
                    
        if best_json:
            return best_json.strip()
        
    except Exception as e:
        eval_logger.debug(f"Error in brace matching: {e}")
    
    # Pattern 3: Pure JSON (entire response)
    if response.startswith('{') and response.endswith('}'):
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError:
            pass
    
    # Pattern 4: Fallback - use regex for simple JSON
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    if matches:
        # Try to parse each match, return first valid one
        for match in sorted(matches, key=len, reverse=True):
            try:
                json.loads(match)
                return match.strip()
            except json.JSONDecodeError:
                continue
    
    # Failed to extract - return empty structure
    eval_logger.warning(f"Failed to extract JSON from response: {response[:100]}...")
    return '{"opinions": []}'


def parse_and_validate_json(json_str: str) -> Dict:
    """
    Parse JSON string and validate basic SemEval structure.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed dict or {} if invalid
    """
    try:
        data = json.loads(json_str)
        
        # Must be a dict
        if not isinstance(data, dict):
            return {}
        
        # Must have opinions (sent_id and text optional in output)
        if "opinions" not in data:
            return {}
        
        if not isinstance(data["opinions"], list):
            return {}
        
        return data
        
    except json.JSONDecodeError as e:
        eval_logger.debug(f"JSON decode error: {e}")
        return {}
    except Exception as e:
        eval_logger.debug(f"Validation error: {e}")
        return {}


def compute_text_spans(text: str, phrases: List[str]) -> List[str]:
    """
    Find character offsets using existing extract_position().
    """
    return [extract_position(text, phrase) for phrase in phrases if phrase]


def normalize_opinion_format(doc_text: str, opinion: Dict) -> Dict:
    """
    Normalize opinion to SemEval format with [texts, spans].
    
    Args:
        doc_text: Original document text
        opinion: Opinion dict (may have various formats)
        
    Returns:
        Normalized opinion with [texts, spans] for each component
    """
    normalized = {}
    
    for key in ["Source", "Target", "Polar_expression"]:
        if key not in opinion:
            normalized[key] = [[], []]
            continue
        
        value = opinion[key]
        
        # Case 1: Already in [texts, spans] format
        if isinstance(value, list) and len(value) == 2:
            if isinstance(value[0], list) and isinstance(value[1], list):
                # Extract texts from old format, then re-compute spans
                phrases = [p for p in value[0] if isinstance(p, str) and p.strip()]
                spans = compute_text_spans(doc_text, phrases)
                normalized[key] = [phrases, spans]
                continue
        
        # Case 2: Simple list of text phrases
        if isinstance(value, list):
            # Filter out non-string items
            phrases = [p for p in value if isinstance(p, str) and p.strip()]
            spans = compute_text_spans(doc_text, phrases)
            normalized[key] = [phrases, spans]
        
        # Case 3: Single string
        elif isinstance(value, str) and value.strip():
            phrases = [value]
            spans = compute_text_spans(doc_text, phrases)
            normalized[key] = [phrases, spans]
        
        # Case 4: Empty/None
        else:
            normalized[key] = [[], []]
    
    valid_polarities = ["Positive", "Negative", "Neutral"]
    polarity = opinion.get("Polarity", "")
    normalized["Polarity"] = polarity if polarity in valid_polarities else ""

    # Copy and validate intensity
    valid_intensities = ["Strong", "Standard", "Weak"]
    intensity = opinion.get("Intensity", "")
    normalized["Intensity"] = intensity if intensity in valid_intensities else ""
    
    return normalized


def convert_to_semeval_format(doc: Dict, model_output: str) -> Dict:
    """
    Convert model output to SemEval format.
    
    Args:
        doc: Original document dict
        model_output: Raw model response string
        
    Returns:
        Dict in SemEval format with sent_id, text, and opinions
    """
    # Extract JSON from response
    json_str = extract_json_from_response(model_output)
    
    # Parse JSON
    parsed = parse_and_validate_json(json_str)
    
    if not parsed or "opinions" not in parsed:
        # Return empty result
        return {
            "sent_id": doc["sent_id"],
            "text": doc["text"],
            "opinions": []
        }
    
    # Normalize each opinion to [texts, spans] format
    normalized_opinions = []
    for opinion in parsed.get("opinions", []):
        try:
            normalized = normalize_opinion_format(doc["text"], opinion)
            normalized_opinions.append(normalized)
        except Exception as e:
            eval_logger.debug(f"Error normalizing opinion: {e}")
            continue
    
    return {
        "sent_id": doc.get("sent_id", parsed.get("sent_id")),
        "text": doc.get("text", parsed.get("text")),
        "opinions": normalized_opinions
    }


# ==================== PROCESS RESULTS ====================

def semeval_process_results(doc: Dict, results: List[str]) -> Dict[str, Any]:
    """
    Process model results for a single document.
    
    This function is called by the framework for each document after
    filtering. It converts model output to metrics that will be aggregated.
    
    Args:
        doc: Original document dict with gold labels
        results: List of model responses (after filtering, usually len=1)
        
    Returns:
        Dict with metric values:
        - semeval_f1: (doc, pred) tuple for aggregation
        - exact_match: 0 or 1
        - valid_json: 0 or 1
        
    Note:
        Framework will aggregate these values across all documents using
        the aggregation functions defined in metric_list.
    """
    # Get filtered response (should be JSON string after extract_json filter)
    response = results[0] if results else ""
    
    # Convert to SemEval format
    pred = convert_to_semeval_format(doc, response)
    
    # Metric 1: Valid JSON rate
    has_predictions = len(pred["opinions"]) > 0
    valid_json = 1 if has_predictions else 0
    
    # Metric 2: Exact match (strict comparison)
    # Note: This is very strict - rarely matches in practice
    exact_match = 1 if pred["opinions"] == doc["opinions"] else 0
    
    # Metric 3: SemEval F1 (needs aggregation)
    # Pass both doc and pred for tuple-based evaluation
    semeval_tuple = (doc, pred)
    
    return {
        "semeval_f1": semeval_tuple,
        "exact_match": exact_match,
        "valid_json": valid_json,
    }


# ==================== AGGREGATION FUNCTIONS ====================

@register_aggregation("semeval_f1_agg")
def semeval_f1_aggregation(items: List[Tuple[Dict, Dict]]) -> float:
    """
    Aggregate SemEval F1 scores across all documents.
    
    Uses the official SemEval-2022 Task 10 evaluation script which:
    - Converts opinions to token-based tuples (holder, target, expression, polarity)
    - Computes weighted overlap-based F1 score
    
    Args:
        items: List of (doc, pred) tuples from all documents
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    if not SEMEVAL_AVAILABLE:
        eval_logger.error("SemEval evaluation module not available!")
        return 0.0
    
    if not items:
        return 0.0
    
    # Build gold and pred dicts keyed by sent_id
    gold_dict = {}
    pred_dict = {}
    
    for doc, pred in items:
        sent_id = doc["sent_id"]
        
        try:
            # Convert to tuple format expected by evaluate.py
            # Each opinion becomes: (holder_tokens, target_tokens, exp_tokens, polarity)
            gold_dict[sent_id] = convert_opinion_to_tuple(doc)
            pred_dict[sent_id] = convert_opinion_to_tuple(pred)
        except Exception as e:
            eval_logger.debug(f"Error converting sent_id {sent_id}: {e}")
            # Add empty tuples for failed conversions
            gold_dict[sent_id] = []
            pred_dict[sent_id] = []
    
    # Validate same keys
    if set(gold_dict.keys()) != set(pred_dict.keys()):
        eval_logger.warning("Gold and pred have different sent_ids!")
    
    # Calculate F1 using official SemEval metric
    try:
        f1 = tuple_f1(
            gold=gold_dict,
            pred=pred_dict,
            keep_polarity=True,
            weighted=True
        )
        
        # Also compute precision and recall for logging
        prec = tuple_precision(gold_dict, pred_dict, keep_polarity=True, weighted=True)
        rec = tuple_recall(gold_dict, pred_dict, keep_polarity=True, weighted=True)
        
        eval_logger.info(f"SemEval Metrics - P: {prec:.4f}, R: {rec:.4f}, F1: {f1:.4f}")
        
        return f1
        
    except Exception as e:
        eval_logger.error(f"Error calculating F1: {e}")
        return 0.0


@register_aggregation("semeval_precision_agg")
def semeval_precision_aggregation(items: List[Tuple[Dict, Dict]]) -> float:
    """Aggregate precision scores (optional metric)."""
    if not SEMEVAL_AVAILABLE or not items:
        return 0.0
    
    gold_dict = {}
    pred_dict = {}
    
    for doc, pred in items:
        sent_id = doc["sent_id"]
        try:
            gold_dict[sent_id] = convert_opinion_to_tuple(doc)
            pred_dict[sent_id] = convert_opinion_to_tuple(pred)
        except:
            gold_dict[sent_id] = []
            pred_dict[sent_id] = []
    
    try:
        prec = tuple_precision(gold_dict, pred_dict, keep_polarity=True, weighted=True)
        return prec
    except:
        return 0.0


@register_aggregation("semeval_recall_agg")
def semeval_recall_aggregation(items: List[Tuple[Dict, Dict]]) -> float:
    """Aggregate recall scores (optional metric)."""
    if not SEMEVAL_AVAILABLE or not items:
        return 0.0
    
    gold_dict = {}
    pred_dict = {}
    
    for doc, pred in items:
        sent_id = doc["sent_id"]
        try:
            gold_dict[sent_id] = convert_opinion_to_tuple(doc)
            pred_dict[sent_id] = convert_opinion_to_tuple(pred)
        except:
            gold_dict[sent_id] = []
            pred_dict[sent_id] = []
    
    try:
        rec = tuple_recall(gold_dict, pred_dict, keep_polarity=True, weighted=True)
        return rec
    except:
        return 0.0


# ==================== HELPER FUNCTIONS ====================

def get_fewshot_samples(n: int = 5) -> List[Dict]:
    """
    Get first N samples from examples pool for few-shot.
    
    This function can be used in fewshot_config.samples if needed,
    but typically framework will auto-select from fewshot_split.
    
    Args:
        n: Number of samples to return
        
    Returns:
        List of example dicts
    """
    # This is placeholder - framework handles auto-selection
    # Only needed if manually specifying samples
    return []


def format_opinion_for_display(opinion: Dict) -> str:
    """
    Format opinion as readable string for few-shot examples.
    
    Args:
        opinion: Opinion dict
        
    Returns:
        Formatted JSON string
    """
    # Simplify for display - only show text spans, not offsets
    display = {
        "Source": opinion.get("Source", [[]])[0],  # Just texts
        "Target": opinion.get("Target", [[]])[0],
        "Polar_expression": opinion.get("Polar_expression", [[]])[0],
        "Polarity": opinion.get("Polarity", "Neutral"),
        "Intensity": opinion.get("Intensity", "Standard")
    }
    return json.dumps(display, ensure_ascii=False, indent=2)


# ==================== DEBUGGING UTILITIES ====================

def validate_example_pool():
    """
    Validate that CoT examples are properly formatted.
    
    Called during task initialization to catch errors early.
    """
    examples = load_fewshot_cot_examples()
    
    for i, ex in enumerate(examples):
        assert "text" in ex, f"Example {i}: missing 'text'"
        assert "reasoning" in ex, f"Example {i}: missing 'reasoning'"
        assert "output" in ex, f"Example {i}: missing 'output'"
        assert "opinions" in ex["output"], f"Example {i}: missing 'opinions' in output"
    
    eval_logger.info(f"✅ Validated {len(examples)} CoT examples")
    return True


# ==================== EXPORT ====================

__all__ = [
    # Data processing
    "process_docs",
    "load_fewshot_cot_examples",
    
    # JSON extraction
    "ExtractJSONFilter",
    "extract_json_from_response",
    "parse_and_validate_json",
    
    # Format conversion
    "compute_text_spans",
    "normalize_opinion_format",
    "convert_to_semeval_format",
    
    # Metrics
    "semeval_process_results",
    "semeval_f1_aggregation",
    "semeval_precision_aggregation",
    "semeval_recall_aggregation",
    
    # Helpers
    "get_fewshot_samples",
    "format_opinion_for_display",
    "validate_example_pool",
]




