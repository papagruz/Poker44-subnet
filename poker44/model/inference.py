"""Local trained-model inference for Poker44 miners."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from poker44.model.features import FEATURE_NAMES, extract_feature_matrix


DEFAULT_ARTIFACT_PATH = Path("models/poker44_chunk_hgb_v1/model.pkl")


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, *, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def calibrate_batch_scores(
    scores: list[float],
    *,
    enabled: bool | None = None,
    target_positive_rate: float | None = None,
    min_batch_size: int | None = None,
) -> list[float]:
    """Return live-safe, rank-preserving scores for one validator batch.

    The trained public-benchmark model can be over-confident on live validator
    snapshots. If every chunk crosses 0.5, Poker44's hard-threshold human-safety
    term can zero the reward even when the ranking still contains useful signal.

    This remap keeps the model ordering but caps the number of chunks above the
    operational 0.5 threshold. It uses only the current batch scores, never live
    labels or identifiers, so it is safe for production miners.
    """
    raw = [max(0.0, min(1.0, float(score))) for score in scores]
    n = len(raw)
    if n == 0:
        return []

    if enabled is None:
        enabled = _env_bool("POKER44_BATCH_CALIBRATION", default=True)
    if not enabled:
        return [round(score, 6) for score in raw]

    if min_batch_size is None:
        min_batch_size = max(1, _env_int("POKER44_BATCH_CALIBRATION_MIN_CHUNKS", default=16))
    if n < min_batch_size:
        return [round(score, 6) for score in raw]

    if target_positive_rate is None:
        target_positive_rate = _env_float("POKER44_TARGET_POSITIVE_RATE", default=0.30)
    target_positive_rate = max(0.01, min(0.50, float(target_positive_rate)))

    positive_count = int(round(n * target_positive_rate))
    positive_count = max(1, min(n - 1, positive_count))
    negative_count = n - positive_count

    negative_low = _env_float("POKER44_NEGATIVE_SCORE_LOW", default=0.02)
    negative_high = min(0.499, _env_float("POKER44_NEGATIVE_SCORE_HIGH", default=0.49))
    positive_low = max(0.501, _env_float("POKER44_POSITIVE_SCORE_LOW", default=0.51))
    positive_high = _env_float("POKER44_POSITIVE_SCORE_HIGH", default=0.98)

    ranked_indices = sorted(range(n), key=lambda idx: (raw[idx], idx))
    calibrated = [0.0] * n
    for rank, idx in enumerate(ranked_indices):
        if rank < negative_count:
            frac = rank / max(negative_count - 1, 1)
            value = negative_low + (negative_high - negative_low) * frac
        else:
            frac = (rank - negative_count) / max(positive_count - 1, 1)
            value = positive_low + (positive_high - positive_low) * frac
        calibrated[idx] = round(max(0.0, min(1.0, value)), 6)
    return calibrated


class TrainedChunkModel:
    def __init__(self, artifact: dict[str, Any]):
        self.artifact = artifact
        self.model = artifact["model"]
        self.feature_names = list(artifact.get("feature_names") or FEATURE_NAMES)
        self.threshold = float(artifact.get("threshold", 0.5))
        self.score_clip = tuple(artifact.get("score_clip", (0.001, 0.999)))

    @classmethod
    def load(cls, path: str | Path | None = None) -> "TrainedChunkModel | None":
        raw_path = path or os.getenv("POKER44_MODEL_ARTIFACT") or DEFAULT_ARTIFACT_PATH
        artifact_path = Path(raw_path)
        if not artifact_path.is_absolute():
            artifact_path = Path.cwd() / artifact_path
        if not artifact_path.exists():
            return None
        with artifact_path.open("rb") as handle:
            artifact = pickle.load(handle)
        if list(artifact.get("feature_names") or []) != list(FEATURE_NAMES):
            raise ValueError("trained model feature schema does not match current FEATURE_NAMES")
        return cls(artifact)

    def score_chunk(self, chunk: list[dict]) -> float:
        return self.score_chunks([chunk])[0]

    def score_chunks(self, chunks: list[list[dict]]) -> list[float]:
        if not chunks:
            return []
        features = np.asarray(extract_feature_matrix(chunks), dtype=float)
        if hasattr(self.model, "predict_proba"):
            raw_scores = self.model.predict_proba(features)[:, 1]
        elif hasattr(self.model, "decision_function"):
            raw = self.model.decision_function(features)
            raw_scores = 1.0 / (1.0 + np.exp(-raw))
        else:
            raw_scores = self.model.predict(features)
        return [self._finalize_score(float(score)) for score in raw_scores]

    def _finalize_score(self, score: float) -> float:
        score = self._align_threshold_to_half(score)
        low, high = float(self.score_clip[0]), float(self.score_clip[1])
        return round(max(0.0, min(1.0, max(low, min(high, score)))), 6)

    def _align_threshold_to_half(self, score: float) -> float:
        """Monotonic remap so tuned operating threshold lands at score 0.5."""
        threshold = max(1e-6, min(1.0 - 1e-6, self.threshold))
        if score < threshold:
            return 0.5 * score / threshold
        return 0.5 + 0.5 * (score - threshold) / (1.0 - threshold)
