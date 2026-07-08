"""Leakage-safe chunk feature extraction for Poker44 miner models."""

from __future__ import annotations

import math
from collections import Counter
from statistics import fmean, pstdev
from typing import Iterable


ACTION_TYPES = ("fold", "check", "call", "bet", "raise", "all_in")
STREETS = ("preflop", "flop", "turn", "river")
NUMERIC_ACTION_FIELDS = (
    "amount",
    "raise_to",
    "call_to",
    "normalized_amount_bb",
    "pot_before",
    "pot_after",
)


FEATURE_NAMES = [
    "hand_count",
    "mean_actions_per_hand",
    "std_actions_per_hand",
    "min_actions_per_hand",
    "max_actions_per_hand",
    "mean_player_count",
    "std_player_count",
    "mean_max_seats",
    "mean_starting_stack",
    "std_starting_stack",
    "min_starting_stack",
    "max_starting_stack",
    "action_entropy",
    "actor_entropy",
    "unique_actor_share",
    "aggression_rate",
    "passive_rate",
    "zero_amount_share",
    "showdown_rate",
    "hands_reaching_flop_rate",
    "hands_reaching_turn_rate",
    "hands_reaching_river_rate",
    *[f"{action}_rate" for action in ACTION_TYPES],
    *[f"{street}_action_share" for street in STREETS],
    *[f"{field}_{stat}" for field in NUMERIC_ACTION_FIELDS for stat in ("mean", "std", "min", "max", "q25", "q50", "q75")],
    "amount_to_pot_before_mean",
    "amount_to_pot_before_std",
    "amount_to_pot_before_max",
    "pot_growth_mean",
    "pot_growth_std",
    "pot_growth_max",
]


def _safe_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _stats(values: Iterable[float]) -> dict[str, float]:
    xs = [float(x) for x in values if math.isfinite(float(x))]
    if not xs:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0}
    ordered = sorted(xs)
    return {
        "mean": float(fmean(ordered)),
        "std": float(pstdev(ordered)) if len(ordered) > 1 else 0.0,
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "q25": _quantile(ordered, 0.25),
        "q50": _quantile(ordered, 0.50),
        "q75": _quantile(ordered, 0.75),
    }


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(sorted_values[low])
    frac = pos - low
    return float(sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac)


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log(p)
    return float(entropy / max(math.log(max(len(counts), 2)), 1e-12))


def extract_chunk_features(chunk: list[dict]) -> list[float]:
    """Return fixed-order numeric features for one miner-visible chunk group."""
    hands = chunk or []
    hand_count = len(hands)
    actions_per_hand: list[float] = []
    player_counts: list[float] = []
    max_seats_values: list[float] = []
    starting_stacks: list[float] = []
    action_counts: Counter = Counter()
    street_counts: Counter = Counter()
    actor_counts: Counter = Counter()
    numeric_values: dict[str, list[float]] = {field: [] for field in NUMERIC_ACTION_FIELDS}
    amount_to_pot_before: list[float] = []
    pot_growth: list[float] = []
    zero_amount_count = 0
    action_total = 0
    showdown_count = 0
    reaches = Counter()

    for hand in hands:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        metadata = hand.get("metadata") or {}
        outcome = hand.get("outcome") or {}

        actions_per_hand.append(float(len(actions)))
        player_counts.append(float(len(players)))
        max_seats = _safe_float(metadata.get("max_seats"))
        if max_seats is not None:
            max_seats_values.append(max_seats)
        for player in players:
            stack = _safe_float(player.get("starting_stack"))
            if stack is not None:
                starting_stacks.append(stack)
        if bool(outcome.get("showdown")):
            showdown_count += 1

        seen_streets = {str(street).lower() for street in streets if street}
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_total += 1
            action_type = str(action.get("action_type") or "").lower()
            if action_type:
                action_counts[action_type] += 1
            street = str(action.get("street") or "").lower()
            if street:
                street_counts[street] += 1
                seen_streets.add(street)
            actor = action.get("actor_seat")
            if actor is not None:
                actor_counts[str(actor)] += 1

            amount = _safe_float(action.get("amount"))
            if amount is not None and abs(amount) <= 1e-12:
                zero_amount_count += 1
            pot_before = _safe_float(action.get("pot_before"))
            pot_after = _safe_float(action.get("pot_after"))
            if amount is not None and pot_before is not None and pot_before > 1e-12:
                amount_to_pot_before.append(amount / pot_before)
            if pot_before is not None and pot_after is not None:
                pot_growth.append(pot_after - pot_before)
            for field in NUMERIC_ACTION_FIELDS:
                value = _safe_float(action.get(field))
                if value is not None:
                    numeric_values[field].append(value)

        for street in ("flop", "turn", "river"):
            if street in seen_streets:
                reaches[street] += 1

    action_den = max(action_total, 1)
    hand_den = max(hand_count, 1)
    feature_map: dict[str, float] = {
        "hand_count": float(hand_count),
        "mean_actions_per_hand": _stats(actions_per_hand)["mean"],
        "std_actions_per_hand": _stats(actions_per_hand)["std"],
        "min_actions_per_hand": _stats(actions_per_hand)["min"],
        "max_actions_per_hand": _stats(actions_per_hand)["max"],
        "mean_player_count": _stats(player_counts)["mean"],
        "std_player_count": _stats(player_counts)["std"],
        "mean_max_seats": _stats(max_seats_values)["mean"],
        "mean_starting_stack": _stats(starting_stacks)["mean"],
        "std_starting_stack": _stats(starting_stacks)["std"],
        "min_starting_stack": _stats(starting_stacks)["min"],
        "max_starting_stack": _stats(starting_stacks)["max"],
        "action_entropy": _entropy(action_counts),
        "actor_entropy": _entropy(actor_counts),
        "unique_actor_share": len(actor_counts) / action_den,
        "aggression_rate": (action_counts.get("bet", 0) + action_counts.get("raise", 0)) / action_den,
        "passive_rate": (action_counts.get("check", 0) + action_counts.get("call", 0)) / action_den,
        "zero_amount_share": zero_amount_count / action_den,
        "showdown_rate": showdown_count / hand_den,
        "hands_reaching_flop_rate": reaches.get("flop", 0) / hand_den,
        "hands_reaching_turn_rate": reaches.get("turn", 0) / hand_den,
        "hands_reaching_river_rate": reaches.get("river", 0) / hand_den,
    }

    for action in ACTION_TYPES:
        feature_map[f"{action}_rate"] = action_counts.get(action, 0) / action_den
    for street in STREETS:
        feature_map[f"{street}_action_share"] = street_counts.get(street, 0) / action_den
    for field in NUMERIC_ACTION_FIELDS:
        stats = _stats(numeric_values[field])
        for stat, value in stats.items():
            feature_map[f"{field}_{stat}"] = value
    ratio_stats = _stats(amount_to_pot_before)
    feature_map["amount_to_pot_before_mean"] = ratio_stats["mean"]
    feature_map["amount_to_pot_before_std"] = ratio_stats["std"]
    feature_map["amount_to_pot_before_max"] = ratio_stats["max"]
    growth_stats = _stats(pot_growth)
    feature_map["pot_growth_mean"] = growth_stats["mean"]
    feature_map["pot_growth_std"] = growth_stats["std"]
    feature_map["pot_growth_max"] = growth_stats["max"]

    return [float(feature_map.get(name, 0.0)) for name in FEATURE_NAMES]


def extract_feature_matrix(chunks: Iterable[list[dict]]) -> list[list[float]]:
    return [extract_chunk_features(chunk) for chunk in chunks]
