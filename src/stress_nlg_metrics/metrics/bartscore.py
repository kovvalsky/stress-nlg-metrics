"""BARTScore metric."""

from functools import lru_cache

import torch


def _check_lengths(references: list[str], candidates: list[str]) -> None:
    """Check that references and candidates have equal length."""
    if len(references) != len(candidates):
        raise ValueError(
            "references and candidates must have the same length"
        )


@lru_cache(maxsize=None)
def _get_scorer(
    model_name: str = "facebook/bart-large-cnn",
):
    """Load and cache the BARTScore model."""
    from string2string.similarity import BARTScore

    return BARTScore(
        model_name_or_path=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def _compute(
    scorer,
    sources: list[str],
    targets: list[str],
    batch_size: int,
) -> list[float]:
    """Compute BARTScore in one direction."""
    result = scorer.compute(
        sources,
        targets,
        agg="mean",
        batch_size=batch_size,
    )

    return [float(score) for score in result["score"]]


def bartscore(
    references: list[str],
    candidates: list[str],
    batch_size: int = 16,
    direction: str = "bi",
    model_name: str = "facebook/bart-large-cnn",
) -> list[float]:
    """Compute BARTScore.

    Args:
        references: Reference sentences.
        candidates: Candidate sentences.
        batch_size: Inference batch size.
        direction: "bi", "forward", or "backward".
        model_name: Hugging Face BART model.
    """
    _check_lengths(references, candidates)

    scorer = _get_scorer(model_name)

    if direction == "forward":
        return _compute(
            scorer,
            references,
            candidates,
            batch_size,
        )

    if direction == "backward":
        return _compute(
            scorer,
            candidates,
            references,
            batch_size,
        )

    if direction == "bi":
        forward = _compute(
            scorer,
            references,
            candidates,
            batch_size,
        )

        backward = _compute(
            scorer,
            candidates,
            references,
            batch_size,
        )

        return [
            (a + b) / 2
            for a, b in zip(forward, backward, strict=True)
        ]

    raise ValueError(
        "direction must be 'bi', 'forward', or 'backward'"
    )