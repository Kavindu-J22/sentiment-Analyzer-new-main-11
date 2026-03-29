"""
Phase II – Transformer-Based Sarcasm / Irony Detector
Uses cardiffnlp/twitter-roberta-base-irony (RoBERTa) to detect nuanced
customer feedback that VADER misses – especially sarcasm & irony.
"""
import re
import pandas as pd
import numpy as np

# ── Model Configuration ──────────────────────────────────────────────────────
MODEL_NAME = "cardiffnlp/twitter-roberta-base-irony"
MAX_LENGTH = 128   # RoBERTa input cap (Twitter-tuned model)
BATCH_SIZE = 16

# Label mapping for the irony model
IRONY_LABELS = {0: 'non_irony', 1: 'irony'}
IRONY_THRESHOLD = 0.55   # Probability cutoff for flagging as sarcastic


def _preprocess_for_roberta(text: str) -> str:
    """Minimal pre-processing for RoBERTa: preserve punctuation & case."""
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+|www\S+', '@URL', text)   # mask URLs
    text = re.sub(r'@\w+', '@USER', text)             # mask mentions
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:512]   # hard cap


def load_roberta_pipeline():
    """
    Load the HuggingFace pipeline for irony detection.
    Returns None if transformers/torch are unavailable.
    """
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        pipe = pipeline(
            task='text-classification',
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        return pipe
    except Exception as e:
        print(f"[Sarcasm Detector] Model load failed: {e}")
        return None


def detect_single(text: str, pipe) -> dict:
    """Run sarcasm detection on a single review text."""
    if pipe is None:
        return {'irony_prob': 0.0, 'non_irony_prob': 1.0, 'is_sarcastic': False}
    cleaned = _preprocess_for_roberta(text)
    if not cleaned:
        return {'irony_prob': 0.0, 'non_irony_prob': 1.0, 'is_sarcastic': False}
    try:
        results = pipe(cleaned)[0]   # list of {label, score}
        score_map = {r['label']: r['score'] for r in results}
        irony_prob = round(score_map.get('irony', 0.0), 4)
        non_irony_prob = round(score_map.get('non_irony', 1.0), 4)
        return {
            'irony_prob': irony_prob,
            'non_irony_prob': non_irony_prob,
            'is_sarcastic': irony_prob >= IRONY_THRESHOLD,
        }
    except Exception:
        return {'irony_prob': 0.0, 'non_irony_prob': 1.0, 'is_sarcastic': False}


def batch_detect_sarcasm(texts: pd.Series, pipe,
                         batch_size: int = BATCH_SIZE,
                         progress_callback=None) -> pd.DataFrame:
    """
    Run sarcasm detection on a Series of texts in batches.
    progress_callback(current, total) is called after each batch.
    Returns DataFrame with irony_prob, non_irony_prob, is_sarcastic columns.
    """
    records = []
    texts_list = texts.tolist()
    total = len(texts_list)

    for start in range(0, total, batch_size):
        batch = [_preprocess_for_roberta(t) for t in texts_list[start:start + batch_size]]
        batch = [t if t else 'neutral review' for t in batch]
        if pipe is not None:
            try:
                raw = pipe(batch)
                for result in raw:
                    score_map = {r['label']: r['score'] for r in result}
                    irony_prob = round(score_map.get('irony', 0.0), 4)
                    records.append({
                        'irony_prob': irony_prob,
                        'non_irony_prob': round(score_map.get('non_irony', 1.0), 4),
                        'is_sarcastic': irony_prob >= IRONY_THRESHOLD,
                    })
            except Exception:
                for _ in batch:
                    records.append({'irony_prob': 0.0,
                                    'non_irony_prob': 1.0, 'is_sarcastic': False})
        else:
            for _ in batch:
                records.append({'irony_prob': 0.0,
                                'non_irony_prob': 1.0, 'is_sarcastic': False})
        if progress_callback:
            progress_callback(min(start + batch_size, total), total)

    return pd.DataFrame(records)


def get_sarcasm_kpis(df: pd.DataFrame) -> dict:
    """Compute sarcasm-related KPIs from analyzed DataFrame."""
    if 'is_sarcastic' not in df.columns:
        return {}
    sarcastic = df[df['is_sarcastic']]
    return {
        'total_analyzed': len(df),
        'sarcastic_count': int(sarcastic.shape[0]),
        'sarcasm_rate_pct': round(sarcastic.shape[0] / max(len(df), 1) * 100, 1),
        'avg_irony_prob': round(df['irony_prob'].mean(), 4),
        'high_confidence_sarcasm': int((df['irony_prob'] >= 0.75).sum()),
    }

