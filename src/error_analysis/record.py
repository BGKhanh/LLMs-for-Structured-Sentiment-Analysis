# src/error_analysis/record.py
"""Field accessors for one row of `samples_*.jsonl` — taxonomy v2 §2.5 Data Contract."""
import json
from typing import Any, Dict, List, Optional


def raw_response(sample: Dict[str, Any]) -> str:
    resps = sample.get("resps") or []
    return resps[0][0] if resps and resps[0] else ""


def filtered_response(sample: Dict[str, Any]) -> str:
    filtered = sample.get("filtered_resps") or []
    return filtered[0] if filtered else ""


def gen_args(sample: Dict[str, Any]) -> Dict[str, Any]:
    return sample.get("arguments", {}).get("gen_args_0", {}).get("arg_1", {}) or {}


def prompt_text(sample: Dict[str, Any]) -> str:
    return sample.get("arguments", {}).get("gen_args_0", {}).get("arg_0", "") or ""


def doc(sample: Dict[str, Any]) -> Dict[str, Any]:
    return sample.get("doc", {}) or {}


def doc_text(sample: Dict[str, Any]) -> str:
    return doc(sample).get("text", "")


def doc_sent_id(sample: Dict[str, Any]) -> str:
    return str(doc(sample).get("sent_id", ""))


def doc_dataset(sample: Dict[str, Any]) -> str:
    return doc(sample).get("dataset", "")


def doc_language(sample: Dict[str, Any]) -> str:
    return doc(sample).get("language", "vi")


def gold_opinions(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = doc(sample).get("opinions_json", "[]")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def predicted_opinions(filtered: str) -> Optional[List[Dict[str, Any]]]:
    """None means "unparseable" (distinct from a validly-parsed empty list)."""
    try:
        parsed = json.loads(filtered)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    opinions = parsed.get("opinions")
    return opinions if isinstance(opinions, list) else []


def load_samples_jsonl(path: str) -> List[Dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples
