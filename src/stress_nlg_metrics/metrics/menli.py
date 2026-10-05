"""MENLI-style NLI metric."""

from functools import lru_cache

import torch


DEFAULT_MODEL = "microsoft/deberta-large-mnli"


def _check_lengths(references: list[str], candidates: list[str]) -> None:
    """Check that references and candidates have equal length."""
    if len(references) != len(candidates):
        raise ValueError(
            "references and candidates must have the same length"
        )


@lru_cache(maxsize=None)
def _get_model(model_name: str = DEFAULT_MODEL):
    """Load and cache tokenizer and NLI model."""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


@torch.no_grad()
def _entailment_prob_batch(
    premises: list[str],
    hypotheses: list[str],
    batch_size: int = 16,
    model_name: str = DEFAULT_MODEL,
) -> list[float]:
    """Return entailment probabilities for premise-hypothesis pairs."""
    _check_lengths(premises, hypotheses)

    tokenizer, model, device = _get_model(model_name)

    scores = []

    for start in range(0, len(premises), batch_size):
        premise_batch = premises[start:start + batch_size]
        hypothesis_batch = hypotheses[start:start + batch_size]

        inputs = tokenizer(
            premise_batch,
            hypothesis_batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)

        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)

        # microsoft/deberta-large-mnli:
        # 0 = contradiction
        # 1 = neutral
        # 2 = entailment
        scores.extend(probabilities[:, 2].tolist())

    return scores


def menli(
    references: list[str],
    candidates: list[str],
    direction: str = "bi",
    batch_size: int = 16,
    model_name: str = DEFAULT_MODEL,
) -> list[float]:
    """Compute a MENLI-style entailment score.

    Args:
        references: Reference sentences.
        candidates: Candidate sentences.
        direction: "bi", "forward", or "backward".
        batch_size: Inference batch size.
        model_name: Hugging Face NLI model.
    """
    _check_lengths(references, candidates)

    if direction == "forward":
        return _entailment_prob_batch(
            references,
            candidates,
            batch_size=batch_size,
            model_name=model_name,
        )

    if direction == "backward":
        return _entailment_prob_batch(
            candidates,
            references,
            batch_size=batch_size,
            model_name=model_name,
        )

    if direction == "bi":
        forward = _entailment_prob_batch(
            references,
            candidates,
            batch_size=batch_size,
            model_name=model_name,
        )

        backward = _entailment_prob_batch(
            candidates,
            references,
            batch_size=batch_size,
            model_name=model_name,
        )

        return [
            (a + b) / 2
            for a, b in zip(forward, backward, strict=True)
        ]

    raise ValueError(
        "direction must be 'bi', 'forward', or 'backward'"
    )