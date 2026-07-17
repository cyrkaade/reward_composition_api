from __future__ import annotations

import pytest

from rcomp.config import ConfigError
from rcomp.partiality import estimate_partiality_from_returns


def test_variance_metrics_for_helpful_partial():
    true_returns = [1.0, 2.0, 3.0, 4.0]
    partial_returns = [0.9, 2.1, 2.8, 4.2]

    metrics = estimate_partiality_from_returns(true_returns, partial_returns)

    deltas = [t - p for t, p in zip(true_returns, partial_returns)]
    delta_mean = sum(deltas) / len(deltas)
    expected_var_delta = sum((d - delta_mean) ** 2 for d in deltas) / (len(deltas) - 1)

    assert metrics["var_delta"] == pytest.approx(expected_var_delta)
    assert metrics["var_delta_over_var_true"] == pytest.approx(expected_var_delta / metrics["var_true"])
    assert metrics["delta_easier_than_true"] is True


def test_variance_metrics_for_misleading_partial():
    true_returns = [1.0, 2.0, 3.0, 4.0]
    partial_returns = [4.0, 3.0, 2.0, 1.0]

    metrics = estimate_partiality_from_returns(true_returns, partial_returns)

    assert metrics["var_delta"] > metrics["var_true"]
    assert metrics["delta_easier_than_true"] is False


def test_zero_true_variance_raises():
    with pytest.raises(ConfigError, match="zero variance"):
        estimate_partiality_from_returns([1.0, 1.0], [0.5, 0.7])
