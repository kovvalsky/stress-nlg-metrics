"""AlignScore metric."""

import os
from functools import lru_cache
from pathlib import Path

import torch


def _check_lengths(references: list[str], candidates: list[str]) -> None:
    """Check that references and candidates have equal length."""
    if len(references) != len(candidates):
        raise ValueError(
            "references and candidates must have the same length"
        )


def _resolve_checkpoint(
    checkpoint_path: str | Path | None,
) -> str:
    """Resolve the AlignScore checkpoint path."""
    if checkpoint_path is None:
        checkpoint_path = os.environ.get(
            "ALIGNSCORE_CHECKPOINT"
        )

    if checkpoint_path is None:
        raise ValueError(
            "AlignScore checkpoint not specified. Pass "
            "checkpoint_path=... or set ALIGNSCORE_CHECKPOINT."
        )

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"AlignScore checkpoint not found: {checkpoint_path}"
        )

    return str(checkpoint_path)


@lru_cache(maxsize=None)
def _get_scorer(
    checkpoint_path: str,
    model_name: str,
    batch_size: int,
    evaluation_mode: str,
):
    """Load and cache an AlignScore scorer."""
    from alignscore import AlignScore

    return AlignScore(
        model=model_name,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ckpt_path=checkpoint_path,
        evaluation_mode=evaluation_mode,
    )


def alignscore(
    references: list[str],
    candidates: list[str],
    direction: str = "bi",
    batch_size: int = 16,
    checkpoint_path: str | Path | None = None,
    model_name: str = "roberta-large",
    evaluation_mode: str = "nli_sp",
) -> list[float]:
    """Compute AlignScore.

    Args:
        references: Reference sentences.
        candidates: Candidate sentences.
        direction: "bi", "forward", or "backward".
        batch_size: AlignScore batch size.
        checkpoint_path: AlignScore checkpoint.
        model_name: Base model used by AlignScore.
        evaluation_mode: AlignScore evaluation mode.
    """
    _check_lengths(references, candidates)

    checkpoint_path = _resolve_checkpoint(checkpoint_path)

    scorer = _get_scorer(
        checkpoint_path,
        model_name,
        batch_size,
        evaluation_mode,
    )

    if direction == "forward":
        return [
            float(score)
            for score in scorer.score(
                contexts=references,
                claims=candidates,
            )
        ]

    if direction == "backward":
        return [
            float(score)
            for score in scorer.score(
                contexts=candidates,
                claims=references,
            )
        ]

    if direction == "bi":
        forward = scorer.score(
            contexts=references,
            claims=candidates,
        )

        backward = scorer.score(
            contexts=candidates,
            claims=references,
        )

        return [
            (float(a) + float(b)) / 2
            for a, b in zip(
                forward,
                backward,
                strict=True,
            )
        ]

    raise ValueError(
        "direction must be 'bi', 'forward', or 'backward'"
    )