from poker44.model.features import FEATURE_NAMES, extract_chunk_features


def _sample_chunk():
    return [
        {
            "hand_id": "hidden-identity-a",
            "metadata": {"max_seats": 6},
            "players": [
                {"player_uid": "seat_1", "seat": 1, "starting_stack": 4.5},
                {"player_uid": "seat_2", "seat": 2, "starting_stack": 4.7},
            ],
            "streets": ["preflop", "flop"],
            "actions": [
                {
                    "action_id": "1",
                    "street": "preflop",
                    "actor_seat": 1,
                    "action_type": "call",
                    "amount": 0.2,
                    "normalized_amount_bb": 10,
                    "pot_before": 0.3,
                    "pot_after": 0.5,
                },
                {
                    "action_id": "2",
                    "street": "flop",
                    "actor_seat": 2,
                    "action_type": "raise",
                    "amount": 0.5,
                    "normalized_amount_bb": 25,
                    "pot_before": 0.5,
                    "pot_after": 1.0,
                },
            ],
            "outcome": {},
        }
    ]


def test_feature_vector_schema_is_fixed_width():
    features = extract_chunk_features(_sample_chunk())

    assert len(features) == len(FEATURE_NAMES)
    assert all(isinstance(value, float) for value in features)


def test_identifier_fields_do_not_affect_features():
    first = _sample_chunk()
    second = _sample_chunk()
    second[0]["hand_id"] = "hidden-identity-b"
    second[0]["actions"][0]["action_id"] = "999"

    assert extract_chunk_features(first) == extract_chunk_features(second)


def test_visible_amount_field_affects_features():
    first = _sample_chunk()
    second = _sample_chunk()
    second[0]["actions"][0]["normalized_amount_bb"] = 100

    assert extract_chunk_features(first) != extract_chunk_features(second)
