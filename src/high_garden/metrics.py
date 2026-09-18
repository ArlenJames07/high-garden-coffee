from __future__ import annotations

import numpy as np


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        raise ValueError("WAPE is undefined when the absolute target sum is zero")
    return float(np.abs(y_true - y_pred).sum() / denominator)
