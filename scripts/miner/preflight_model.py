#!/usr/bin/env python3
"""Pre-registration model preflight for the Poker44 miner."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

from neurons.miner import Miner
from poker44.model.inference import DEFAULT_ARTIFACT_PATH
from poker44.score.scoring import reward
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)


DEFAULT_BASE_URL = "https://api.poker44.net/api/v1/benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Poker44 miner model contract, manifest, benchmark sample, and latency."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--source-date", default="", help="Benchmark sourceDate. Defaults to latest.")
    parser.add_argument("--limit", type=int, default=24, help="Chunk rows to fetch per page.")
    parser.add_argument("--pages", type=int, default=1, help="Maximum benchmark pages to fetch.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds.")
    parser.add_argument(
        "--fail-on-opaque-manifest",
        action="store_true",
        help="Exit non-zero when manifest is not transparent.",
    )
    parser.add_argument(
        "--allow-benchmark-failure",
        action="store_true",
        help="Keep manifest/contract preflight successful when public benchmark API is unavailable.",
    )
    return parser.parse_args()


def fetch_benchmark(args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    session = requests.Session()
    source_date = args.source_date
    if not source_date:
        status = session.get(args.base_url, timeout=args.timeout)
        status.raise_for_status()
        source_date = status.json()["data"]["latestSourceDate"]

    rows: list[dict[str, Any]] = []
    cursor = None
    for _ in range(max(1, args.pages)):
        params: dict[str, Any] = {"sourceDate": source_date, "limit": args.limit}
        if cursor:
            params["cursor"] = cursor
        response = session.get(f"{args.base_url}/chunks", params=params, timeout=args.timeout)
        response.raise_for_status()
        data = response.json()["data"]
        rows.extend(data.get("chunks") or [])
        cursor = data.get("nextCursor")
        if not cursor:
            break
    return source_date, rows


def build_manifest(repo_root: Path) -> dict[str, Any]:
    artifact_path = Path(os.getenv("POKER44_MODEL_ARTIFACT") or DEFAULT_ARTIFACT_PATH)
    if not artifact_path.is_absolute():
        artifact_path = repo_root / artifact_path
    artifact_sha = ""
    if artifact_path.exists():
        digest = hashlib.sha256()
        with artifact_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        artifact_sha = digest.hexdigest()
    model_loaded = artifact_path.exists()
    return build_local_model_manifest(
        repo_root=repo_root,
        implementation_files=[
            repo_root / "neurons" / "miner.py",
            repo_root / "poker44" / "model" / "features.py",
            repo_root / "poker44" / "model" / "inference.py",
        ],
        defaults={
            "model_name": "poker44-public-benchmark-hgb" if model_loaded else "poker44-reference-heuristic",
            "model_version": "1",
            "framework": "scikit-learn-hist-gradient-boosting" if model_loaded else "python-heuristic",
            "license": "MIT",
            "repo_url": "" if model_loaded else "https://github.com/Poker44/Poker44-subnet",
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
                "Trained on public Poker44 benchmark releases only, using miner-visible chunk features."
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


def score_rows(rows: list[dict[str, Any]]) -> tuple[list[float], list[int], list[float]]:
    scores: list[float] = []
    labels: list[int] = []
    latencies_ms: list[float] = []
    trained_model = Miner._get_trained_model()
    for row in rows:
        chunks = row.get("chunks") or []
        ground_truth = row.get("groundTruth") or []
        if len(chunks) != len(ground_truth):
            raise ValueError(
                f"contract mismatch in row {row.get('chunkId', '<unknown>')}: "
                f"chunks={len(chunks)} labels={len(ground_truth)}"
            )
        if trained_model is not None:
            started = time.perf_counter()
            row_scores = trained_model.score_chunks(chunks)
            per_chunk_latency = ((time.perf_counter() - started) * 1000.0) / max(len(chunks), 1)
        else:
            row_scores = []
            for chunk in chunks:
                started = time.perf_counter()
                row_scores.append(Miner.score_chunk(chunk))
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
            per_chunk_latency = 0.0
        for score, label in zip(row_scores, ground_truth):
            if trained_model is not None:
                latencies_ms.append(per_chunk_latency)
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"score out of range: {score}")
            scores.append(float(score))
            labels.append(int(label))
    return scores, labels, latencies_ms


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    if not os.getenv("POKER44_MODEL_REPO_COMMIT"):
        try:
            repo_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            repo_commit = ""
        if repo_commit:
            os.environ["POKER44_MODEL_REPO_COMMIT"] = repo_commit

    manifest = build_manifest(repo_root)
    compliance = evaluate_manifest_compliance(manifest)
    print("manifest_status", compliance["status"])
    print("manifest_missing_fields", compliance["missing_fields"])
    print("manifest_policy_violations", compliance["policy_violations"])
    print("manifest_digest", manifest_digest(manifest))
    print("implementation_sha256", manifest.get("implementation_sha256", ""))

    if args.fail_on_opaque_manifest and compliance["status"] != "transparent":
        return 2

    Miner._get_trained_model()

    try:
        source_date, rows = fetch_benchmark(args)
    except requests.RequestException as exc:
        print("benchmark_error", repr(exc))
        if args.allow_benchmark_failure:
            sample_chunks = [
                [
                    {
                        "players": [{}, {}],
                        "streets": ["preflop"],
                        "actions": [{"action_type": "call"}, {"action_type": "check"}],
                        "outcome": {},
                    }
                ],
                [
                    {
                        "players": [{}, {}],
                        "streets": ["preflop", "flop", "turn"],
                        "actions": [{"action_type": "call"}] * 10,
                        "outcome": {"showdown": True},
                    }
                ],
            ]
            sample_scores = [Miner.score_chunk(chunk) for chunk in sample_chunks]
            print("contract_smoke_scores", sample_scores)
            print("contract_smoke_predictions", [score >= 0.5 for score in sample_scores])
            return 0
        return 3
    scores, labels, latencies_ms = score_rows(rows)
    if not scores:
        raise RuntimeError("benchmark returned no scoreable examples")

    value, metrics = reward(np.asarray(scores, float), np.asarray(labels, int))
    print("source_date", source_date)
    print("rows", len(rows))
    print("examples", len(scores))
    print("reward", round(value, 6))
    for key in sorted(metrics):
        print(key, round(float(metrics[key]), 6))
    print("score_min", round(min(scores), 6))
    print("score_max", round(max(scores), 6))
    print("score_mean", round(statistics.fmean(scores), 6))
    print("positive_rate_at_0_5", round(statistics.fmean(s >= 0.5 for s in scores), 6))
    print("latency_ms_mean", round(statistics.fmean(latencies_ms), 6))
    print("latency_ms_p95", round(float(np.percentile(latencies_ms, 95)), 6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
