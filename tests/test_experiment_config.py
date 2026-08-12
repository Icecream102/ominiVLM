import json

import pytest

from scripts.run_experiment import config_environment, load_config


def valid_config():
    return {
        "pretrain": {"epochs": 2, "batch_size": 128, "learning_rate": 4e-4},
        "sft": {"epochs": 2, "batch_size": 64, "learning_rate": 5e-6},
        "grpo": {
            "samples": 5000,
            "group_size": 4,
            "ppo_epochs": 2,
            "learning_rate": 1e-6,
            "beta": 0.02,
            "adaptive_beta": True,
            "target_kl": 0.1,
            "kl_stop": 1.0,
        },
        "evaluation": {"samples": 500, "conditions": ["correct", "black", "shuffled"]},
    }


def test_load_config_and_map_environment(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(valid_config()), encoding="utf-8")
    config = load_config(path)
    environment = config_environment(config, base={})
    assert environment["GRPO_ADAPTIVE_BETA"] == "1"
    assert environment["SFT_BATCH_SIZE"] == "64"
    assert environment["PIPELINE_SEED"] == "42"
    assert environment["EVAL_CONDITIONS"] == "correct black shuffled"


def test_config_rejects_invalid_target_kl(tmp_path):
    config = valid_config()
    config["grpo"]["target_kl"] = 0
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="target_kl"):
        load_config(path)
