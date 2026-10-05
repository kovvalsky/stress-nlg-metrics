_scorer = None


def _get_scorer():
    """Load BERTScore model only when first needed."""
    global _scorer

    if _scorer is None:
        import torch
        from bert_score import BERTScorer

        _scorer = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            lang="en",
            rescale_with_baseline=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=16,
        )

        _scorer._tokenizer.model_max_length = 512

    return _scorer


def bertscore(
    references: list[str],
    candidates: list[str],
    direction="bi",
) -> list[float]:
    """Compute bidirectional BERTScore."""
    scorer = _get_scorer()

    if direction == "bi":
        _, _, cr = scorer.score(candidates, references, verbose=False,)
        _, _, rc = scorer.score(references, candidates, verbose=False,)

        return [(a + b) / 2 for a, b in zip(cr.tolist(), rc.tolist(), strict=True,)]

    if direction == "forward":
        _, _, rc = scorer.score(references, candidates, verbose=False,)
        return rc.tolist()
    if direction == "backward":
        _, _, cr = scorer.score(candidates, references, verbose=False,)
        return cr.tolist()
    else:
        raise ValueError(f"Invalid direction: {direction}")