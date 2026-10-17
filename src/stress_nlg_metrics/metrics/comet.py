"""COMET metric."""

from functools import lru_cache

import torch


DEFAULT_MODEL = "Unbabel/wmt22-comet-da"


def _check_lengths(references: list[str], candidates: list[str]) -> None:
    """Check that references and candidates have equal length."""
    if len(references) != len(candidates):
        raise ValueError(
            "references and candidates must have the same length"
        )


@lru_cache(maxsize=None)
def _get_model(model_name: str = DEFAULT_MODEL):
    """Download/load and cache a COMET model."""
    from comet import download_model, load_from_checkpoint

    model_path = download_model(model_name)

    return load_from_checkpoint(model_path)


def comet(
    references: list[str],
    candidates: list[str],
    batch_size: int = 16,
    model_name: str = DEFAULT_MODEL,
    src_tr: str = ""
) -> list[float]:
    """Compute COMET scores using the current experimental setup.

    The candidate is used as both src and mt, matching the original
    notebook implementation.
    """
    _check_lengths(references, candidates)

    model = _get_model(model_name)

    def detect_src(src, ref, cand):
        if src == "": return ""
        if src == "reference": return ref
        if src == "candidate": return cand
        raise ValueError(
                    f"unknown value for src_tr parameter: {src_tr}"
                )

    samples = [
        {
            "src": detect_src(src_tr, reference, candidate),
            "mt": candidate,
            "ref": reference,
        }
        for reference, candidate in zip(
            references,
            candidates,
            strict=True,
        )
    ]

    gpus = 1 if torch.cuda.is_available() else 0

    output = model.predict(
        samples,
        batch_size=batch_size,
        gpus=gpus,
    )

    return [float(score) for score in output.scores]