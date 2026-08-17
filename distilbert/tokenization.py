"""
DistilBERT Tokenization
-------------------------
Tokenizes light-cleaned text for DistilBERT. Moved out of the TF-IDF
feature engineering module, since the two model paths are separate.
"""

import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_bert_tokenized_features(texts: pd.Series, model_name: str = "distilbert-base-uncased",
                                   max_length: int = 128):
    """
    Tokenize light-cleaned text for DistilBERT.

    Requires the `transformers` library:
        pip install transformers torch

    Returns a dict with 'input_ids' and 'attention_mask' tensors, ready
    to feed into a DistilBERT model.
    """
    try:
        from transformers import DistilBertTokenizerFast
    except ImportError as e:
        raise ImportError(
            "The 'transformers' package is required for BERT tokenization. "
            "Install it with: pip install transformers torch"
        ) from e

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    logger.info(f"Tokenized {len(texts)} rows for {model_name} "
                f"(max_length={max_length})")
    return encoded
