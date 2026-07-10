"""Poker44 miner with trained local model and heuristic fallback."""

# from __future__ import annotations

import time
from collections import Counter
import hashlib
import os
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.model.inference import DEFAULT_ARTIFACT_PATH, TrainedChunkModel, calibrate_batch_scores
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


class Miner(BaseMinerNeuron):
    """
    Local chunk-level bot detector.

    It uses a trained public-benchmark model when an artifact is available and
    falls back to deterministic behavioral heuristics otherwise.
    """

    _trained_model: TrainedChunkModel | None = None
    _trained_model_checked = False

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.trained_model = self._get_trained_model()
        model_loaded = self.trained_model is not None
        if self.trained_model is not None:
            self._warm_trained_model()
        bt.logging.info(
            "🤖 Poker44 Miner started "
            f"mode={'trained-model' if model_loaded else 'heuristic-fallback'}"
        )
        repo_root = Path(__file__).resolve().parents[1]
        artifact_path = self._artifact_path(repo_root)
        artifact_sha = self._sha256_file(artifact_path) if artifact_path.exists() else ""
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[
                Path(__file__).resolve(),
                repo_root / "poker44" / "model" / "features.py",
                repo_root / "poker44" / "model" / "inference.py",
            ],
            defaults={
                "model_name": "poker44-public-benchmark-hgb" if model_loaded else "poker44-reference-heuristic",
                "model_version": "1",
                "framework": "scikit-learn-hist-gradient-boosting" if model_loaded else "python-heuristic",
                "license": "MIT",
                "repo_url": "https://github.com/papagruz/Poker44-subnet" if model_loaded else "https://github.com/Poker44/Poker44-subnet",
                "artifact_url": str(artifact_path.relative_to(repo_root)) if model_loaded else "",
                "artifact_sha256": artifact_sha,
                "notes": (
                    "Local sklearn model trained only on public Poker44 benchmark releases."
                    if model_loaded
                    else "Reference heuristic miner shipped with the Poker44 subnet."
                ),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained on public Poker44 benchmark releases only, using miner-visible "
                    "chunk features from players, streets, actions, and outcome. No identifiers, "
                    "hashes, source dates, labels, or validator-private fields are used as features."
                    if model_loaded
                    else "Reference heuristic miner. No training step. Uses only runtime chunk features."
                ),
                "training_data_sources": ["Poker44 public benchmark API"] if model_loaded else ["none"],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data or private labels."
                    if model_loaded
                    else "This reference miner does not train on validator-only evaluation data."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        
        # # Attach handlers after initialization
        # self.axon.attach(
        #     forward_fn = self.forward,
        #     blacklist_fn = self.blacklist,
        #     priority_fn = self.priority,
        # )
        # bt.logging.info("Attaching forward function to miner axon.")
        
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    def _warm_trained_model(self) -> None:
        """Run one tiny inference to avoid first validator request paying sklearn setup cost."""
        try:
            assert self.trained_model is not None
            self.trained_model.score_chunks(
                [
                    [
                        {
                            "players": [{}, {}],
                            "streets": ["preflop"],
                            "actions": [
                                {"action_type": "call", "street": "preflop", "amount": 0.02},
                                {"action_type": "check", "street": "preflop", "amount": 0.0},
                            ],
                            "outcome": {},
                        }
                    ]
                ]
            )
            bt.logging.info("Trained Poker44 model warmup complete.")
        except Exception as exc:
            bt.logging.warning(f"Trained Poker44 model warmup failed: {exc}")

    @classmethod
    def _get_trained_model(cls) -> TrainedChunkModel | None:
        if cls._trained_model_checked:
            return cls._trained_model
        cls._trained_model_checked = True
        try:
            cls._trained_model = TrainedChunkModel.load()
        except Exception as exc:
            bt.logging.warning(f"Unable to load trained Poker44 model artifact; using heuristic fallback: {exc}")
            cls._trained_model = None
        return cls._trained_model

    @staticmethod
    def _artifact_path(repo_root: Path) -> Path:
        raw = os.getenv("POKER44_MODEL_ARTIFACT") or str(DEFAULT_ARTIFACT_PATH)
        path = Path(raw)
        return path if path.is_absolute() else repo_root / path

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b=""):
                digest.update(chunk)
        return digest.hexdigest()

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one deterministic bot-risk score per chunk."""
        chunks = synapse.chunks or []
        if self.trained_model is not None:
            raw_scores = self.trained_model.score_chunks(chunks)
            scores = calibrate_batch_scores(raw_scores)
        else:
            raw_scores = [self.score_chunk(chunk) for chunk in chunks]
            scores = calibrate_batch_scores(raw_scores)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Miner predictions: {synapse.predictions}")
        self._log_score_distribution(raw_scores=raw_scores, scores=scores)
        bt.logging.info(
            f"Scored {len(chunks)} chunks with "
            f"{'trained model' if self.trained_model is not None else 'heuristic fallback'}."
        )
        return synapse

    @staticmethod
    def _score_stats(scores: list[float]) -> dict[str, float]:
        if not scores:
            return {
                "min": 0.0,
                "mean": 0.0,
                "max": 0.0,
                "positive_count": 0,
                "positive_rate": 0.0,
            }
        positive_count = sum(1 for score in scores if score >= 0.5)
        return {
            "min": min(scores),
            "mean": sum(scores) / len(scores),
            "max": max(scores),
            "positive_count": positive_count,
            "positive_rate": positive_count / len(scores),
        }

    @classmethod
    def _log_score_distribution(cls, *, raw_scores: list[float], scores: list[float]) -> None:
        raw = cls._score_stats(raw_scores)
        calibrated = cls._score_stats(scores)
        bt.logging.info(
            "Risk score stats | "
            f"raw_min={raw['min']:.6f} raw_mean={raw['mean']:.6f} raw_max={raw['max']:.6f} "
            f"raw_positive={raw['positive_count']}/{len(raw_scores)} raw_positive_rate={raw['positive_rate']:.3f} | "
            f"final_min={calibrated['min']:.6f} final_mean={calibrated['mean']:.6f} final_max={calibrated['max']:.6f} "
            f"final_positive={calibrated['positive_count']}/{len(scores)} final_positive_rate={calibrated['positive_rate']:.3f}"
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter(action.get("action_type") for action in actions)
        meaningful_actions = max(
            1,
            sum(
                action_counts.get(kind, 0)
                for kind in ("call", "check", "bet", "raise", "fold")
            ),
        )

        call_ratio = action_counts.get("call", 0) / meaningful_actions
        check_ratio = action_counts.get("check", 0) / meaningful_actions
        fold_ratio = action_counts.get("fold", 0) / meaningful_actions
        raise_ratio = action_counts.get("raise", 0) / meaningful_actions
        street_depth = len(streets) / 3.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0

        player_count_signal = 0.0
        if players:
            player_count_signal = (6 - min(len(players), 6)) / 4.0

        score = 0.0
        score += 0.32 * street_depth
        score += 0.22 * showdown_flag
        score += 0.18 * cls._clamp01(call_ratio / 0.35)
        score += 0.12 * cls._clamp01(check_ratio / 0.30)
        score += 0.08 * cls._clamp01(player_count_signal)
        score -= 0.18 * cls._clamp01(fold_ratio / 0.55)
        score -= 0.10 * cls._clamp01(raise_ratio / 0.20)

        return cls._clamp01(score)

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        trained_model = cls._get_trained_model()
        if trained_model is not None:
            return trained_model.score_chunk(chunk)
        if not chunk:
            return 0.5

        hand_scores = [cls._score_hand(hand) for hand in chunk]
        avg_score = sum(hand_scores) / len(hand_scores)

        return round(cls._clamp01(avg_score), 6)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Random miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
