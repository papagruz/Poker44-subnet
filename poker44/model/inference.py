"""Local trained-model inference for Poker44 miners."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from poker44.model.features import FEATURE_NAMES, extract_feature_matrix


DEFAULT_ARTIFACT_PATH = Path("models/poker44_chunk_hgb_v1/model.pkl")


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
