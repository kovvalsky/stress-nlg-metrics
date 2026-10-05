"""Surface-form NLG evaluation metrics."""

from functools import lru_cache


def _check_lengths(references: list[str], candidates: list[str]) -> None:
    """Check that references and candidates have equal length."""
    if len(references) != len(candidates):
        raise ValueError(
            "references and candidates must have the same length"
        )


@lru_cache(maxsize=1)
def _get_bleu():
    """Create the SacreBLEU sentence-level BLEU scorer."""
    from sacrebleu.metrics import BLEU

    return BLEU(effective_order=True)


@lru_cache(maxsize=1)
def _get_chrf():
    """Create the SacreBLEU chrF scorer."""
    from sacrebleu.metrics import CHRF

    return CHRF()


@lru_cache(maxsize=1)
def _get_rouge():
    """Create the ROUGE scorer."""
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )


@lru_cache(maxsize=1)
def _prepare_nltk() -> None:
    """Download NLTK resources required by METEOR."""
    import nltk

    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)


def bleu(
    references: list[str],
    candidates: list[str],
) -> list[float]:
    """Compute sentence-level BLEU for corresponding sentence pairs."""
    _check_lengths(references, candidates)
    scorer = _get_bleu()

    return [
        scorer.sentence_score(candidate, [reference]).score
        for reference, candidate in zip(
            references,
            candidates,
            strict=True,
        )
    ]


def chrf(
    references: list[str],
    candidates: list[str],
) -> list[float]:
    """Compute sentence-level chrF for corresponding sentence pairs."""
    _check_lengths(references, candidates)
    scorer = _get_chrf()

    return [
        scorer.sentence_score(candidate, [reference]).score
        for reference, candidate in zip(
            references,
            candidates,
            strict=True,
        )
    ]


def rouge(
    references: list[str],
    candidates: list[str],
    mode: str = "rougeL",
) -> list[float]:
    """Compute ROUGE F1 scores.

    Args:
        references: Reference sentences.
        candidates: Candidate sentences.
        mode: One of "rouge1", "rouge2", or "rougeL".
    """
    _check_lengths(references, candidates)

    if mode not in {"rouge1", "rouge2", "rougeL"}:
        raise ValueError(
            "mode must be one of: rouge1, rouge2, rougeL"
        )

    scorer = _get_rouge()

    return [
        scorer.score(reference, candidate)[mode].fmeasure
        for reference, candidate in zip(
            references,
            candidates,
            strict=True,
        )
    ]


def meteor(
    references: list[str],
    candidates: list[str],
) -> list[float]:
    """Compute METEOR for corresponding sentence pairs."""
    _check_lengths(references, candidates)
    _prepare_nltk()

    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import single_meteor_score

    scores = []

    for reference, candidate in zip(
        references,
        candidates,
        strict=True,
    ):
        reference_tokens = word_tokenize(reference)
        candidate_tokens = word_tokenize(candidate)

        scores.append(
            single_meteor_score(
                reference_tokens,
                candidate_tokens,
            )
        )

    return scores