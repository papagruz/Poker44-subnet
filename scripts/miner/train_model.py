#!/usr/bin/env python3
"""Train a leakage-safe Poker44 chunk classifier on public benchmark releases."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss

from poker44.model.features import FEATURE_NAMES, extract_feature_matrix
from poker44.score.scoring import reward
from poker44.validator.payload_view import build_miner_payload_hand


BASE_URL = "https://api.poker44.net/api/v1/benchmark"
MODEL_DIR = Path("models/poker44_chunk_hgb_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Poker44 chunk model from public benchmark data.")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--release-limit", type=int, default=8)
    parser.add_argument("--holdout-releases", type=int, default=2)
    parser.add_argument("--limit", type=int, default=24, help="Rows per API page.")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--cache-dir", default="data/poker44_benchmark")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--max-pages-per-release", type=int, default=0, help="0 means all pages.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--download-method", choices=("curl", "requests"), default="curl")
    parser.add_argument(
        "--no-sanitized-augmentation",
        action="store_true",
        help="Disable adding miner-visible sanitized copies of public benchmark chunks to training and calibration.",
    )
    parser.add_argument("--max-iter", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.20)
    return parser.parse_args()


def request_json(
    session: requests.Session,
    url: str,
    *,
    timeout: float,
    method: str = "requests",
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if method == "curl":
        full_url = url
        if params:
            full_url = f"{url}?{urlencode(params)}"
        command = [
            "curl",
            "-4",
            "-L",
            "--compressed",
            "--connect-timeout",
            "20",
            "--max-time",
            str(int(timeout)),
            "--silent",
            "--show-error",
            full_url,
        ]
        output = subprocess.check_output(command, text=True)
        return json.loads(output)

    response = session.get(
        url,
        timeout=timeout,
        headers={"Accept-Encoding": "gzip, deflate"},
        params=params,
    )
    response.raise_for_status()
    return response.json()


def fetch_releases(args: argparse.Namespace) -> list[str]:
    session = requests.Session()
    payload = request_json(
        session,
        f"{args.base_url}/releases",
        timeout=args.timeout,
        method=args.download_method,
        params={"limit": args.release_limit},
    )
    releases = payload["data"]["releases"]
    return [str(item["sourceDate"]) for item in releases]


def download_release(args: argparse.Namespace, source_date: str, cache_dir: Path) -> Path:
    target = cache_dir / f"{source_date}.json"
    if args.skip_download and target.exists():
        return target

    session = requests.Session()
    rows: list[dict[str, Any]] = []
    cursor = None
    pages = 0
    while True:
        params: dict[str, Any] = {"sourceDate": source_date, "limit": args.limit}
        if cursor:
            params["cursor"] = cursor
        started = time.time()
        payload = request_json(
            session,
            f"{args.base_url}/chunks",
            timeout=args.timeout,
            method=args.download_method,
            params=params,
        )
        data = payload["data"]
        new_rows = data.get("chunks") or []
        rows.extend(new_rows)
        pages += 1
        print(f"downloaded sourceDate={source_date} page={pages} rows={len(new_rows)} total={len(rows)} elapsed={time.time() - started:.1f}s")
        cursor = data.get("nextCursor")
        if not cursor:
            break
        if args.max_pages_per_release and pages >= args.max_pages_per_release:
            break

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"sourceDate": source_date, "rows": rows}), encoding="utf-8")
    return target


def load_examples(
    paths: list[Path],
    *,
    include_sanitized: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    chunks: list[list[dict]] = []
    labels: list[int] = []
    release_ids: list[str] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_date = str(payload["sourceDate"])
        for row in payload.get("rows") or []:
            row_chunks = row.get("chunks") or []
            row_labels = row.get("groundTruth") or []
            if len(row_chunks) != len(row_labels):
                raise ValueError(f"contract mismatch in {source_date}: chunks={len(row_chunks)} labels={len(row_labels)}")
            chunks.extend(row_chunks)
            labels.extend(int(x) for x in row_labels)
            release_ids.extend([source_date] * len(row_labels))
            if include_sanitized:
                sanitized_chunks = [
                    [build_miner_payload_hand(hand) for hand in chunk]
                    for chunk in row_chunks
                ]
                chunks.extend(sanitized_chunks)
                labels.extend(int(x) for x in row_labels)
                release_ids.extend([source_date] * len(row_labels))
    return np.asarray(extract_feature_matrix(chunks), dtype=float), np.asarray(labels, dtype=int), release_ids


def metrics_for(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    value, details = reward(scores, labels)
    result = {key: float(value) for key, value in details.items()}
    result["reward"] = float(value)
    if len(set(labels.tolist())) > 1:
        result["average_precision"] = float(average_precision_score(labels, scores))
        result["brier"] = float(brier_score_loss(labels, scores))
    else:
        result["average_precision"] = 0.0
        result["brier"] = 0.0
    return result


def align_threshold_to_half(scores: np.ndarray, threshold: float) -> np.ndarray:
    threshold = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    scores = np.asarray(scores, dtype=float)
    lower = 0.5 * scores / threshold
    upper = 0.5 + 0.5 * (scores - threshold) / (1.0 - threshold)
    return np.clip(np.where(scores < threshold, lower, upper), 0.0, 1.0)


def choose_threshold(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.05) -> float:
    negatives = scores[labels == 0]
    positives = scores[labels == 1]
    if len(negatives) == 0 or len(positives) == 0:
        return 0.5
    candidates = sorted(set(float(x) for x in scores), reverse=True)
    best_threshold = 0.5
    best_recall = -1.0
    for threshold in candidates:
        hard = scores >= threshold
        fpr = float(np.mean(hard[labels == 0]))
        recall = float(np.mean(hard[labels == 1]))
        if fpr <= max_fpr and recall > best_recall:
            best_recall = recall
            best_threshold = threshold
    return float(best_threshold)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = repo_root / args.cache_dir
    model_dir = repo_root / args.model_dir
    releases = fetch_releases(args)
    if len(releases) < 3:
        raise RuntimeError("need at least 3 releases for date-split training")
    print("releases_newest_first", releases)

    paths = [download_release(args, source_date, cache_dir) for source_date in releases]
    holdout_count = max(1, min(args.holdout_releases, len(paths) - 2))
    holdout_paths = paths[:holdout_count]
    calibration_paths = paths[holdout_count : holdout_count + 1]
    train_paths = paths[holdout_count + 1 :]

    include_sanitized = not args.no_sanitized_augmentation
    x_train, y_train, _ = load_examples(train_paths, include_sanitized=include_sanitized)
    x_cal, y_cal, _ = load_examples(calibration_paths, include_sanitized=include_sanitized)
    x_holdout, y_holdout, holdout_release_ids = load_examples(holdout_paths)
    x_holdout_sanitized, y_holdout_sanitized, holdout_release_ids_sanitized = load_examples(
        holdout_paths,
        include_sanitized=True,
    )
    print({"train_examples": int(len(y_train)), "cal_examples": int(len(y_cal)), "holdout_examples": int(len(y_holdout))})

    base_model = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        random_state=44,
    )
    base_model.fit(x_train, y_train)
    model = base_model

    cal_scores = model.predict_proba(x_cal)[:, 1]
    threshold = choose_threshold(cal_scores, y_cal, max_fpr=0.05)
    holdout_scores_raw = model.predict_proba(x_holdout)[:, 1]
    holdout_scores = align_threshold_to_half(holdout_scores_raw, threshold)
    all_metrics = metrics_for(holdout_scores, y_holdout)
    holdout_sanitized_scores_raw = model.predict_proba(x_holdout_sanitized)[:, 1]
    holdout_sanitized_scores = align_threshold_to_half(holdout_sanitized_scores_raw, threshold)
    all_metrics_sanitized = metrics_for(holdout_sanitized_scores, y_holdout_sanitized)
    by_release: dict[str, dict[str, float]] = {}
    for release in sorted(set(holdout_release_ids)):
        mask = np.asarray([item == release for item in holdout_release_ids], dtype=bool)
        by_release[release] = metrics_for(holdout_scores[mask], y_holdout[mask])
    by_release_sanitized: dict[str, dict[str, float]] = {}
    for release in sorted(set(holdout_release_ids_sanitized)):
        mask = np.asarray([item == release for item in holdout_release_ids_sanitized], dtype=bool)
        by_release_sanitized[release] = metrics_for(
            holdout_sanitized_scores[mask],
            y_holdout_sanitized[mask],
        )

    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "schema_version": "chunk-features-v1",
        "threshold": threshold,
        "score_clip": (0.001, 0.999),
        "sanitized_augmentation": include_sanitized,
        "model_params": {
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "max_leaf_nodes": args.max_leaf_nodes,
            "l2_regularization": args.l2_regularization,
            "random_state": 44,
        },
        "train_releases": [json.loads(path.read_text(encoding="utf-8"))["sourceDate"] for path in train_paths],
        "calibration_releases": [json.loads(path.read_text(encoding="utf-8"))["sourceDate"] for path in calibration_paths],
        "holdout_releases": [json.loads(path.read_text(encoding="utf-8"))["sourceDate"] for path in holdout_paths],
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "model.pkl").open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (model_dir / "feature_schema.json").write_text(json.dumps({"feature_names": FEATURE_NAMES}, indent=2), encoding="utf-8")
    (model_dir / "metrics_by_release.json").write_text(
        json.dumps(
            {
                "holdout": all_metrics,
                "holdout_sanitized_augmented_view": all_metrics_sanitized,
                "by_release": by_release,
                "by_release_sanitized_augmented_view": by_release_sanitized,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (model_dir / "train_config.json").write_text(json.dumps({k: v for k, v in vars(args).items()}, indent=2, sort_keys=True), encoding="utf-8")
    print("saved", model_dir / "model.pkl")
    print("threshold", threshold)
    print("holdout_metrics", json.dumps(all_metrics, sort_keys=True))
    print("holdout_sanitized_augmented_view_metrics", json.dumps(all_metrics_sanitized, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
