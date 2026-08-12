import pytest

from evaluation.grpo_rewards import (
    reference_reward,
    repetition_ratio,
    unigram_f1,
    update_kl_beta,
)


def test_exact_reference_scores_higher_than_unrelated_text():
    exact, _ = reference_reward("a red bus on the road", ["a red bus on the road"])
    unrelated, _ = reference_reward("two cats sleeping indoors", ["a red bus on the road"])
    assert exact > unrelated
    assert exact > 0.99


def test_empty_prediction_receives_negative_reward():
    reward, components = reference_reward("", ["a valid answer"])
    assert reward == -1.0
    assert components["rouge_l"] == 0.0


def test_repetition_penalty_detects_repeated_bigrams():
    assert repetition_ratio(["a", "b", "c"]) == 0.0
    assert repetition_ratio(["a", "b", "a", "b"]) > 0.0


def test_unigram_f1_counts_duplicate_tokens():
    assert unigram_f1(["cat", "cat"], ["cat"]) == pytest.approx(2 / 3)


def test_custom_reward_weights_are_applied():
    default, _ = reference_reward("a red bus", ["a red bus"])
    length_only, _ = reference_reward(
        "a red bus", ["a red bus"],
        weights={
            "unigram_f1": 0.0,
            "rouge_l": 0.0,
            "meteor_exact": 0.0,
            "length_score": 0.5,
            "repetition_penalty": 0.0,
        },
    )
    assert default > length_only
    assert length_only == pytest.approx(0.5)


def test_kl_controller_moves_beta_toward_target():
    assert update_kl_beta(0.02, observed_kl=0.5, target_kl=0.1) > 0.02
    assert update_kl_beta(0.02, observed_kl=0.01, target_kl=0.1) < 0.02
    assert update_kl_beta(0.02, observed_kl=0.1, target_kl=0.1) == pytest.approx(0.02)


def test_kl_controller_validates_target():
    with pytest.raises(ValueError):
        update_kl_beta(0.02, observed_kl=0.1, target_kl=0.0)
