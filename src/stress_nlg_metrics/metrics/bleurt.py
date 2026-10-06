"""BLEURT metric."""

import os
from functools import lru_cache
from pathlib import Path


def _check_lengths(references: list[str], candidates: list[str]) -> None:
    """Check that references and candidates have equal length."""
    if len(references) != len(candidates):
        raise ValueError(
            "references and candidates must have the same length"
        )


def _resolve_checkpoint(
    checkpoint_path: str | Path | None,
) -> str:
    """Resolve the BLEURT checkpoint path.

    The path can be passed directly or supplied through the
    BLEURT_CHECKPOINT environment variable.
    """
    if checkpoint_path is None:
        checkpoint_path = os.environ.get("BLEURT_CHECKPOINT")

    if checkpoint_path is None:
        raise ValueError(
            "BLEURT checkpoint not specified. Pass "
            "checkpoint_path=... or set BLEURT_CHECKPOINT."
        )

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"BLEURT checkpoint not found: {checkpoint_path}"
        )

    return str(checkpoint_path)


@lru_cache(maxsize=None)
def _get_scorer(checkpoint_path: str):
    """Load and cache a BLEURT scorer."""
    from bleurt import score as bleurt_score

    return bleurt_score.BleurtScorer(checkpoint_path)


def bleurt(
    references: list[str],
    candidates: list[str],
    direction: str = "bi",
    batch_size: int = 16,
    checkpoint_path: str | Path | None = None,
) -> list[float]:
    """Compute BLEURT scores.

    Args:
        references: Reference sentences.
        candidates: Candidate sentences.
        direction: "bi", "forward", or "backward".
        batch_size: BLEURT batch size.
        checkpoint_path: Path to the BLEURT checkpoint.
    """
    _check_lengths(references, candidates)

    checkpoint_path = _resolve_checkpoint(checkpoint_path)
    scorer = _get_scorer(checkpoint_path)

    if direction == "forward":
        return scorer.score(
            references=references,
            candidates=candidates,
            batch_size=batch_size,
        )

    if direction == "backward":
        return scorer.score(
            references=candidates,
            candidates=references,
            batch_size=batch_size,
        )

    if direction == "bi":
        forward = scorer.score(
            references=references,
            candidates=candidates,
            batch_size=batch_size,
        )

        backward = scorer.score(
            references=candidates,
            candidates=references,
            batch_size=batch_size,
        )

        return [
            (a + b) / 2
            for a, b in zip(forward, backward, strict=True)
        ]

    raise ValueError(
        "direction must be 'bi', 'forward', or 'backward'"
    )