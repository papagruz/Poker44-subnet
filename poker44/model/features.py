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

PER_HAND_RATE_FIELDS = (
    "fold",
    "check",
    "call",
    "bet",
    "raise",
    "aggression",
    "passive",
    "unique_actor_share",
    "actions_per_player",
    "street_depth",
)

STREET_ACTION_TYPES = ("fold", "check", "call", "bet", "raise")
FIRST_LAST_ACTION_TYPES = ("fold", "check", "call", "bet", "raise")


EXTRA_FEATURE_NAMES = [
    *[
        f"per_hand_{field}_{stat}"
        for field in PER_HAND_RATE_FIELDS
        for stat in ("mean", "std", "q25", "q50", "q75")
    ],
    *[
        f"{street}_{action}_rate"
        for street in STREETS
        for action in STREET_ACTION_TYPES
    ],
    "preflop_aggression_rate",
    "postflop_aggression_rate",
    "turn_river_call_rate",
    "river_fold_rate",
    *[f"first_action_{action}_share" for action in FIRST_LAST_ACTION_TYPES],
    *[f"last_action_{action}_share" for action in FIRST_LAST_ACTION_TYPES],
    "hero_action_share",
    "hero_fold_rate",
    "hero_check_rate",
    "hero_call_rate",
    "hero_aggression_rate",
    "hero_preflop_action_share",
    "first_actor_is_hero_rate",
    "last_actor_is_hero_rate",
    "max_actor_action_share",
    "actor_action_count_std",
    "actor_action_count_cv",
    "stack_spread_mean",
    "stack_spread_std",
    "short_stack_share_mean",
    "deep_stack_share_mean",
    "small_amount_share",
    "medium_amount_share",
    "large_amount_share",
    "overbet_amount_share",
    "amount_to_pot_small_share",
    "amount_to_pot_medium_share",
    "amount_to_pot_large_share",
    "raise_to_call_to_ratio_mean",
    "raise_to_call_to_ratio_std",
    "call_to_pot_before_mean",
    "call_to_pot_before_std",
    "bet_raise_amount_to_pot_mean",
    "bet_raise_amount_to_pot_std",
]


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
    *EXTRA_FEATURE_NAMES,
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


def _street_name(value: object) -> str:
    if isinstance(value, dict):
        value = value.get("street")
    return str(value or "").lower()


def _street_depth(streets: Iterable[object], actions: Iterable[dict]) -> float:
    order = {"preflop": 1, "flop": 2, "turn": 3, "river": 4}
    depth = 0
    for street in streets:
        depth = max(depth, order.get(_street_name(street), 0))
    for action in actions:
        if isinstance(action, dict):
            depth = max(depth, order.get(_street_name(action.get("street")), 0))
    return float(depth)


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
    per_hand_values: dict[str, list[float]] = {field: [] for field in PER_HAND_RATE_FIELDS}
    street_action_counts: Counter = Counter()
    first_action_counts: Counter = Counter()
    last_action_counts: Counter = Counter()
    actor_total_counts: Counter = Counter()
    hero_action_count = 0
    hero_action_counts: Counter = Counter()
    hero_preflop_action_count = 0
    first_actor_is_hero_count = 0
    last_actor_is_hero_count = 0
    stack_spreads: list[float] = []
    short_stack_shares: list[float] = []
    deep_stack_shares: list[float] = []
    amount_positive_count = 0
    amount_small_count = 0
    amount_medium_count = 0
    amount_large_count = 0
    amount_overbet_count = 0
    amount_to_pot_small_count = 0
    amount_to_pot_medium_count = 0
    amount_to_pot_large_count = 0
    raise_to_call_to_ratios: list[float] = []
    call_to_pot_before_values: list[float] = []
    bet_raise_amount_to_pot: list[float] = []
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
        hero_seat = _safe_float(metadata.get("hero_seat"))
        hero_seat_i = int(hero_seat) if hero_seat is not None else 0

        actions_per_hand.append(float(len(actions)))
        player_counts.append(float(len(players)))
        max_seats = _safe_float(metadata.get("max_seats"))
        if max_seats is not None:
            max_seats_values.append(max_seats)
        for player in players:
            stack = _safe_float(player.get("starting_stack"))
            if stack is not None:
                starting_stacks.append(stack)
        hand_stacks = [
            _safe_float(player.get("starting_stack"))
            for player in players
            if isinstance(player, dict)
        ]
        hand_stacks = [stack for stack in hand_stacks if stack is not None]
        if hand_stacks:
            stack_spreads.append(max(hand_stacks) - min(hand_stacks))
            short_stack_shares.append(sum(stack <= 1.0 for stack in hand_stacks) / len(hand_stacks))
            deep_stack_shares.append(sum(stack >= 5.0 for stack in hand_stacks) / len(hand_stacks))
        if bool(outcome.get("showdown")):
            showdown_count += 1

        seen_streets = {_street_name(street) for street in streets if street}
        hand_action_counts: Counter = Counter()
        hand_actor_counts: Counter = Counter()
        hand_actions = [action for action in actions if isinstance(action, dict)]
        if hand_actions:
            first_action_type = str(hand_actions[0].get("action_type") or "").lower()
            last_action_type = str(hand_actions[-1].get("action_type") or "").lower()
            if first_action_type:
                first_action_counts[first_action_type] += 1
            if last_action_type:
                last_action_counts[last_action_type] += 1
            first_actor = hand_actions[0].get("actor_seat")
            last_actor = hand_actions[-1].get("actor_seat")
            if hero_seat_i and _safe_float(first_actor) == float(hero_seat_i):
                first_actor_is_hero_count += 1
            if hero_seat_i and _safe_float(last_actor) == float(hero_seat_i):
                last_actor_is_hero_count += 1
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_total += 1
            action_type = str(action.get("action_type") or "").lower()
            if action_type:
                action_counts[action_type] += 1
                hand_action_counts[action_type] += 1
            street = _street_name(action.get("street"))
            if street:
                street_counts[street] += 1
                seen_streets.add(street)
                if action_type:
                    street_action_counts[(street, action_type)] += 1
            actor = action.get("actor_seat")
            if actor is not None:
                actor_counts[str(actor)] += 1
                hand_actor_counts[str(actor)] += 1
                actor_total_counts[str(actor)] += 1
            actor_float = _safe_float(actor)
            if hero_seat_i and actor_float == float(hero_seat_i):
                hero_action_count += 1
                if action_type:
                    hero_action_counts[action_type] += 1
                if street == "preflop":
                    hero_preflop_action_count += 1

            amount = _safe_float(action.get("amount"))
            if amount is not None and abs(amount) <= 1e-12:
                zero_amount_count += 1
            normalized_amount_bb = _safe_float(action.get("normalized_amount_bb"))
            if normalized_amount_bb is not None and normalized_amount_bb > 0:
                amount_positive_count += 1
                if normalized_amount_bb <= 1.0:
                    amount_small_count += 1
                elif normalized_amount_bb <= 8.0:
                    amount_medium_count += 1
                elif normalized_amount_bb <= 24.0:
                    amount_large_count += 1
                else:
                    amount_overbet_count += 1
            pot_before = _safe_float(action.get("pot_before"))
            pot_after = _safe_float(action.get("pot_after"))
            if amount is not None and pot_before is not None and pot_before > 1e-12:
                ratio = amount / pot_before
                amount_to_pot_before.append(ratio)
                if ratio <= 0.33:
                    amount_to_pot_small_count += 1
                elif ratio <= 0.75:
                    amount_to_pot_medium_count += 1
                else:
                    amount_to_pot_large_count += 1
                if action_type in {"bet", "raise"}:
                    bet_raise_amount_to_pot.append(ratio)
            if pot_before is not None and pot_after is not None:
                pot_growth.append(pot_after - pot_before)
            raise_to = _safe_float(action.get("raise_to"))
            call_to = _safe_float(action.get("call_to"))
            if raise_to is not None and call_to is not None and call_to > 1e-12:
                raise_to_call_to_ratios.append(raise_to / call_to)
            if call_to is not None and pot_before is not None and pot_before > 1e-12:
                call_to_pot_before_values.append(call_to / pot_before)
            for field in NUMERIC_ACTION_FIELDS:
                value = _safe_float(action.get(field))
                if value is not None:
                    numeric_values[field].append(value)

        hand_action_den = max(sum(hand_action_counts.values()), 1)
        hand_player_den = max(len(players), 1)
        per_hand_values["fold"].append(hand_action_counts.get("fold", 0) / hand_action_den)
        per_hand_values["check"].append(hand_action_counts.get("check", 0) / hand_action_den)
        per_hand_values["call"].append(hand_action_counts.get("call", 0) / hand_action_den)
        per_hand_values["bet"].append(hand_action_counts.get("bet", 0) / hand_action_den)
        per_hand_values["raise"].append(hand_action_counts.get("raise", 0) / hand_action_den)
        per_hand_values["aggression"].append((hand_action_counts.get("bet", 0) + hand_action_counts.get("raise", 0)) / hand_action_den)
        per_hand_values["passive"].append((hand_action_counts.get("check", 0) + hand_action_counts.get("call", 0)) / hand_action_den)
        per_hand_values["unique_actor_share"].append(len(hand_actor_counts) / hand_action_den)
        per_hand_values["actions_per_player"].append(len(hand_actions) / hand_player_den)
        per_hand_values["street_depth"].append(_street_depth(streets, hand_actions))

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

    for field in PER_HAND_RATE_FIELDS:
        stats = _stats(per_hand_values[field])
        for stat in ("mean", "std", "q25", "q50", "q75"):
            feature_map[f"per_hand_{field}_{stat}"] = stats[stat]

    for street in STREETS:
        street_den = max(street_counts.get(street, 0), 1)
        for action in STREET_ACTION_TYPES:
            feature_map[f"{street}_{action}_rate"] = street_action_counts.get((street, action), 0) / street_den

    preflop_den = max(street_counts.get("preflop", 0), 1)
    postflop_total = sum(street_counts.get(street, 0) for street in ("flop", "turn", "river"))
    postflop_den = max(postflop_total, 1)
    turn_river_den = max(street_counts.get("turn", 0) + street_counts.get("river", 0), 1)
    river_den = max(street_counts.get("river", 0), 1)
    feature_map["preflop_aggression_rate"] = (
        street_action_counts.get(("preflop", "bet"), 0)
        + street_action_counts.get(("preflop", "raise"), 0)
    ) / preflop_den
    feature_map["postflop_aggression_rate"] = sum(
        street_action_counts.get((street, action), 0)
        for street in ("flop", "turn", "river")
        for action in ("bet", "raise")
    ) / postflop_den
    feature_map["turn_river_call_rate"] = (
        street_action_counts.get(("turn", "call"), 0)
        + street_action_counts.get(("river", "call"), 0)
    ) / turn_river_den
    feature_map["river_fold_rate"] = street_action_counts.get(("river", "fold"), 0) / river_den

    for action in FIRST_LAST_ACTION_TYPES:
        feature_map[f"first_action_{action}_share"] = first_action_counts.get(action, 0) / hand_den
        feature_map[f"last_action_{action}_share"] = last_action_counts.get(action, 0) / hand_den

    hero_den = max(hero_action_count, 1)
    feature_map["hero_action_share"] = hero_action_count / action_den
    feature_map["hero_fold_rate"] = hero_action_counts.get("fold", 0) / hero_den
    feature_map["hero_check_rate"] = hero_action_counts.get("check", 0) / hero_den
    feature_map["hero_call_rate"] = hero_action_counts.get("call", 0) / hero_den
    feature_map["hero_aggression_rate"] = (hero_action_counts.get("bet", 0) + hero_action_counts.get("raise", 0)) / hero_den
    feature_map["hero_preflop_action_share"] = hero_preflop_action_count / hero_den
    feature_map["first_actor_is_hero_rate"] = first_actor_is_hero_count / hand_den
    feature_map["last_actor_is_hero_rate"] = last_actor_is_hero_count / hand_den

    actor_counts_values = [float(value) for value in actor_total_counts.values()]
    actor_count_stats = _stats(actor_counts_values)
    feature_map["max_actor_action_share"] = (max(actor_counts_values) / action_den) if actor_counts_values else 0.0
    feature_map["actor_action_count_std"] = actor_count_stats["std"]
    feature_map["actor_action_count_cv"] = actor_count_stats["std"] / max(actor_count_stats["mean"], 1e-12)

    stack_spread_stats = _stats(stack_spreads)
    feature_map["stack_spread_mean"] = stack_spread_stats["mean"]
    feature_map["stack_spread_std"] = stack_spread_stats["std"]
    feature_map["short_stack_share_mean"] = _stats(short_stack_shares)["mean"]
    feature_map["deep_stack_share_mean"] = _stats(deep_stack_shares)["mean"]

    amount_den = max(amount_positive_count, 1)
    ratio_den = max(len(amount_to_pot_before), 1)
    feature_map["small_amount_share"] = amount_small_count / amount_den
    feature_map["medium_amount_share"] = amount_medium_count / amount_den
    feature_map["large_amount_share"] = amount_large_count / amount_den
    feature_map["overbet_amount_share"] = amount_overbet_count / amount_den
    feature_map["amount_to_pot_small_share"] = amount_to_pot_small_count / ratio_den
    feature_map["amount_to_pot_medium_share"] = amount_to_pot_medium_count / ratio_den
    feature_map["amount_to_pot_large_share"] = amount_to_pot_large_count / ratio_den
    feature_map["raise_to_call_to_ratio_mean"] = _stats(raise_to_call_to_ratios)["mean"]
    feature_map["raise_to_call_to_ratio_std"] = _stats(raise_to_call_to_ratios)["std"]
    feature_map["call_to_pot_before_mean"] = _stats(call_to_pot_before_values)["mean"]
    feature_map["call_to_pot_before_std"] = _stats(call_to_pot_before_values)["std"]
    feature_map["bet_raise_amount_to_pot_mean"] = _stats(bet_raise_amount_to_pot)["mean"]
    feature_map["bet_raise_amount_to_pot_std"] = _stats(bet_raise_amount_to_pot)["std"]

    return [float(feature_map.get(name, 0.0)) for name in FEATURE_NAMES]


def extract_feature_matrix(chunks: Iterable[list[dict]]) -> list[list[float]]:
    return [extract_chunk_features(chunk) for chunk in chunks]
