from poker44.model.inference import calibrate_batch_scores


def test_batch_calibration_caps_positive_rate_and_preserves_order():
    raw_scores = [0.99 - i * 0.001 for i in range(100)]

    calibrated = calibrate_batch_scores(
        raw_scores,
        enabled=True,
        target_positive_rate=0.30,
        min_batch_size=16,
    )

    assert len(calibrated) == len(raw_scores)
    assert all(0.0 <= score <= 1.0 for score in calibrated)
    assert sum(score >= 0.5 for score in calibrated) == 30

    # Monotonic: higher raw score must not receive lower calibrated score.
    for left, right in zip(calibrated, calibrated[1:]):
        assert left >= right


def test_batch_calibration_default_live_target_is_human_safe(monkeypatch):
    monkeypatch.delenv("POKER44_TARGET_POSITIVE_RATE", raising=False)
    raw_scores = [0.99 - i * 0.001 for i in range(100)]

    calibrated = calibrate_batch_scores(raw_scores, enabled=True, min_batch_size=16)

    assert sum(score >= 0.5 for score in calibrated) == 15


def test_batch_calibration_leaves_small_batches_unchanged():
    raw_scores = [0.91, 0.72, 0.11]

    assert calibrate_batch_scores(
        raw_scores,
        enabled=True,
        target_positive_rate=0.30,
        min_batch_size=16,
    ) == raw_scores


def test_batch_calibration_can_be_disabled():
    raw_scores = [0.9] * 100

    assert calibrate_batch_scores(raw_scores, enabled=False) == raw_scores
