"""PLS regression and PLS-DA, by NIPALS, with the overfitting made visible rather than warned about.

``docs/knowledge/chemometrics.md``: PLS finds latent variables maximising covariance with the
response, assumes a linear relation between spectrum and response, and **overfits fast with many
latent variables**. The number of components is therefore not a hyperparameter to tune quietly on
the test set - it is the model's capacity, and this module makes it inspectable: every fit records
the variance it explains in ``X`` and in ``Y`` per component, so a component that explains a large
share of the spectrum and almost none of the response is visible as what it is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from usis.core.errors import UsisError
from usis.core.types import FloatArray

__all__ = [
    "MIN_USEFUL_Y_VARIANCE",
    "PlsModel",
    "RegressionError",
    "one_hot",
    "pls",
    "pls_da_predict",
]

#: A latent variable contributing less response variance than this is capacity, not signal.
#: A stated convention: 1% of the response is below the noise on any calibration set we hold.
MIN_USEFUL_Y_VARIANCE = 0.01


class RegressionError(UsisError):
    """A regression was asked for something the data cannot support."""


def _two_d(data: FloatArray | np.ndarray, name: str) -> FloatArray:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.size == 0:
        raise RegressionError(f"{name} must be a non-empty 2-D array, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise RegressionError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class PlsModel:
    """A fitted PLS model and the accounting that shows where its capacity went."""

    x_mean: FloatArray
    y_mean: FloatArray
    x_weights: FloatArray
    x_loadings: FloatArray
    y_loadings: FloatArray
    coefficients: FloatArray
    x_variance_explained: FloatArray
    y_variance_explained: FloatArray
    n_components: int
    n_samples: int

    def predict(self, data: FloatArray | np.ndarray) -> FloatArray:
        """Predict the response for new spectra."""
        array = _two_d(data, "data")
        if array.shape[1] != self.x_mean.size:
            raise RegressionError(
                f"model was fitted on {self.x_mean.size} channels, got {array.shape[1]}"
            )
        return np.asarray((array - self.x_mean) @ self.coefficients + self.y_mean, dtype=np.float64)

    @property
    def capacity_warning(self) -> str:
        """Names latent variables that buy no response variance - pure capacity, fitting noise.

        The first version of this property looked for a component explaining a *large* share of
        ``X`` and none of ``Y`` - the classic interferent. Measured on planted problems it almost
        never fires, and the reason is structural rather than a bug: NIPALS orders components by
        covariance with the response, so PLS does not spend a component on ``X``-only variance
        while any covariance remains; the interferent is absorbed into the early components
        instead.

        What *does* happen constantly is components that add nothing at all. Ask for four latent
        variables on a problem with two and the third contributes 0.02% of the response variance.
        That is capacity spent on noise, it is the ordinary way PLS overfits, and it is what this
        reports. The ``X`` share is quoted alongside, because a useless component that *also*
        carries spectral variance is the interferent case on the occasions it does occur.
        """
        useless = [
            index
            for index in range(self.n_components)
            if self.y_variance_explained[index] < MIN_USEFUL_Y_VARIANCE
        ]
        if not useless:
            return ""
        detail = ", ".join(
            f"#{index + 1} (y {self.y_variance_explained[index]:.2%}, "
            f"x {self.x_variance_explained[index]:.2%})"
            for index in useless
        )
        return (
            f"{len(useless)} of {self.n_components} latent variables each explain under "
            f"{MIN_USEFUL_Y_VARIANCE:.0%} of the response variance and are fitting noise: {detail}"
        )


def pls(
    data: FloatArray | np.ndarray,
    response: FloatArray | np.ndarray,
    n_components: int,
    *,
    tolerance: float = 1e-10,
    max_iter: int = 500,
) -> PlsModel:
    """Partial least squares by NIPALS, mean-centred and not scaled.

    Not autoscaled, for the same reason PCA here is not: dividing each channel by its standard
    deviation promotes the empty regions between bands, where the standard deviation is noise, to
    equal footing with the bands.
    """
    x = _two_d(data, "data")
    y = _two_d(response, "response")
    if x.shape[0] != y.shape[0]:
        raise RegressionError(f"data has {x.shape[0]} samples and response has {y.shape[0]}")
    if n_components < 1 or n_components > min(x.shape):
        raise RegressionError(
            f"n_components must be between 1 and min(n_samples, n_channels) = {min(x.shape)}, "
            f"got {n_components}"
        )
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    x_residual, y_residual = x - x_mean, y - y_mean
    x_total = float((x_residual**2).sum()) or 1.0
    y_total = float((y_residual**2).sum()) or 1.0

    weights, x_loadings, y_loadings = [], [], []
    x_explained, y_explained = [], []
    for _ in range(n_components):
        u = y_residual[:, 0].copy()
        w = t = q = np.zeros(1)
        for _ in range(max_iter):
            w = x_residual.T @ u
            norm = float(np.linalg.norm(w))
            if norm == 0.0:
                break
            w = w / norm
            t_new = x_residual @ w
            q = y_residual.T @ t_new
            q_norm = float(np.linalg.norm(q))
            if q_norm == 0.0:
                t = t_new
                break
            u_new = y_residual @ (q / q_norm)
            if np.linalg.norm(u_new - u) <= tolerance * max(float(np.linalg.norm(u)), 1.0):
                t, u = t_new, u_new
                break
            t, u = t_new, u_new
        denominator = float(t @ t)
        if denominator <= 0.0:
            raise RegressionError(
                "a PLS component collapsed to zero variance, which means the requested number of "
                "components exceeds the structure present in the data"
            )
        p = x_residual.T @ t / denominator
        c = y_residual.T @ t / denominator
        before_x, before_y = float((x_residual**2).sum()), float((y_residual**2).sum())
        x_residual = x_residual - np.outer(t, p)
        y_residual = y_residual - np.outer(t, c)
        x_explained.append((before_x - float((x_residual**2).sum())) / x_total)
        y_explained.append((before_y - float((y_residual**2).sum())) / y_total)
        weights.append(w)
        x_loadings.append(p)
        y_loadings.append(c)

    w_matrix = np.column_stack(weights)
    p_matrix = np.column_stack(x_loadings)
    c_matrix = np.column_stack(y_loadings)
    rotation = w_matrix @ np.linalg.pinv(p_matrix.T @ w_matrix)
    return PlsModel(
        x_mean=np.asarray(x_mean, dtype=np.float64),
        y_mean=np.asarray(y_mean, dtype=np.float64),
        x_weights=np.asarray(w_matrix, dtype=np.float64),
        x_loadings=np.asarray(p_matrix, dtype=np.float64),
        y_loadings=np.asarray(c_matrix, dtype=np.float64),
        coefficients=np.asarray(rotation @ c_matrix.T, dtype=np.float64),
        x_variance_explained=np.asarray(x_explained, dtype=np.float64),
        y_variance_explained=np.asarray(y_explained, dtype=np.float64),
        n_components=n_components,
        n_samples=x.shape[0],
    )


def one_hot(labels: list[str], classes: list[str] | None = None) -> tuple[FloatArray, list[str]]:
    """Indicator matrix for PLS-DA, plus the class order it is indexed by."""
    order = sorted(set(labels)) if classes is None else list(classes)
    index = {name: position for position, name in enumerate(order)}
    matrix = np.zeros((len(labels), len(order)), dtype=np.float64)
    for row, label in enumerate(labels):
        if label not in index:
            raise RegressionError(f"label {label!r} is not among the declared classes {order}")
        matrix[row, index[label]] = 1.0
    return matrix, order


def pls_da_predict(
    model: PlsModel, data: FloatArray | np.ndarray, classes: list[str]
) -> tuple[list[str], FloatArray]:
    """Predicted class per spectrum, and the raw indicator scores that produced it.

    The scores are returned rather than hidden because they are **not probabilities**: PLS-DA
    regresses onto an indicator matrix, so a score can exceed one or fall below zero, and two
    classes can both score high. Turning them into a posterior needs a calibration step that this
    baseline deliberately does not invent - the caller decides, and
    :class:`~usis.core.types.BeliefState` will not accept an uncalibrated softmax as though it were
    one.
    """
    scores = model.predict(data)
    if scores.shape[1] != len(classes):
        raise RegressionError(
            f"model predicts {scores.shape[1]} responses but {len(classes)} classes were named"
        )
    predicted = [classes[int(np.argmax(row))] for row in scores]
    return predicted, np.asarray(scores, dtype=np.float64)
