"""Metric registry and convenient evaluation functions."""

from importlib import import_module

from .evaluation import check_metric


# Each metric specifies:
#   module: where the implementation lives
#   function: function to import
#   defaults: default keyword arguments passed to the metric
METRICS = {
    "bleu": {
        "module": "stress_nlg_metrics.metrics.surface",
        "function": "bleu",
        "defaults": {},
    },
    "chrf": {
        "module": "stress_nlg_metrics.metrics.surface",
        "function": "chrf",
        "defaults": {},
    },
    "rouge": {
        "module": "stress_nlg_metrics.metrics.surface",
        "function": "rouge",
        "defaults": {
            "mode": "rougeL",
        },
    },
    "meteor": {
        "module": "stress_nlg_metrics.metrics.surface",
        "function": "meteor",
        "defaults": {},
    },
    "bertscore": {
        "module": "stress_nlg_metrics.metrics.bertscore",
        "function": "bertscore",
        "defaults": {
            "direction": "bi",
        },
    },
    "bleurt": {
        "module": "stress_nlg_metrics.metrics.bleurt",
        "function": "bleurt",
        "defaults": {
            "direction": "bi",
            "batch_size": 16,
        },
    },
    "bartscore": {
        "module": "stress_nlg_metrics.metrics.bartscore",
        "function": "bartscore",
        "defaults": {
            "direction": "bi",
            "batch_size": 16,
        },
    },
    "comet": {
        "module": "stress_nlg_metrics.metrics.comet",
        "function": "comet",
        "defaults": {
            "batch_size": 16,
        },
    },
    "menli": {
        "module": "stress_nlg_metrics.metrics.menli",
        "function": "menli",
        "defaults": {
            "direction": "bi",
            "batch_size": 16,
        },
    },
    "alignscore": {
        "module": "stress_nlg_metrics.metrics.alignscore",
        "function": "alignscore",
        "defaults": {
            "direction": "bi",
            "batch_size": 16,
        },
    },
}


def available_metrics() -> list[str]:
    """Return the names of all registered metrics."""
    return sorted(METRICS)


def get_metric(name: str):
    """Import and return a registered metric function."""
    if name not in METRICS:
        raise ValueError(
            f"Unknown metric: {name}. "
            f"Available metrics: {available_metrics()}"
        )

    spec = METRICS[name]
    module = import_module(spec["module"])

    return getattr(module, spec["function"])


def get_metric_defaults(name: str) -> dict:
    """Return a copy of the default arguments for a metric."""
    if name not in METRICS:
        raise ValueError(
            f"Unknown metric: {name}. "
            f"Available metrics: {available_metrics()}"
        )

    return METRICS[name]["defaults"].copy()


def evaluate(
    name: str,
    data,
    verbose: int = 0,
    **kwargs,
):
    """Evaluate one registered metric.

    Metric-specific keyword arguments override registry defaults.

    Example:
        evaluate("menli", data, direction="forward")
    """
    metric = get_metric(name)

    metric_kwargs = get_metric_defaults(name)
    metric_kwargs.update(kwargs)

    return check_metric(
        metric,
        data,
        verbose=verbose,
        **metric_kwargs,
    )


def evaluate_many(
    names: list[str],
    data,
    metric_kwargs: dict[str, dict] | None = None,
    verbose: int = 0,
) -> dict:
    """Evaluate several metrics and return a results dictionary.

    Args:
        names: Metric names to evaluate.
        data: Stress-test dataset.
        metric_kwargs: Optional metric-specific argument overrides.
            Example:
                {
                    "rouge": {"mode": "rougeL"},
                    "menli": {"direction": "forward"},
                }
        verbose: Verbosity passed to check_metric().

    Returns:
        Dictionary of the form:
            {
                "bleu": {
                    "scores": ...,
                    "fails": ...,
                    "fail_rate": ...,
                },
                ...
            }
    """
    metric_kwargs = metric_kwargs or {}
    results = {}

    for name in names:
        kwargs = metric_kwargs.get(name, {})

        scores, fails, fail_rate = evaluate(
            name,
            data,
            verbose=verbose,
            **kwargs,
        )

        results[name] = {
            "scores": scores,
            "fails": fails,
            "fail_rate": fail_rate,
        }

    return results