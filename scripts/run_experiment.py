"""Validated configuration entry point for the full MiniMind-V pipeline."""

import argparse
import json
import os
import subprocess
from pathlib import Path


ENV_MAP = {
    ("pretrain", "epochs"): "PRETRAIN_EPOCHS",
    ("pretrain", "batch_size"): "PRETRAIN_BATCH_SIZE",
    ("pretrain", "learning_rate"): "PRETRAIN_LR",
    ("sft", "epochs"): "SFT_EPOCHS",
    ("sft", "batch_size"): "SFT_BATCH_SIZE",
    ("sft", "learning_rate"): "SFT_LR",
    ("grpo", "samples"): "GRPO_SAMPLES",
    ("grpo", "group_size"): "GRPO_GROUP_SIZE",
    ("grpo", "ppo_epochs"): "GRPO_PPO_EPOCHS",
    ("grpo", "learning_rate"): "GRPO_LR",
    ("grpo", "beta"): "GRPO_BETA",
    ("grpo", "adaptive_beta"): "GRPO_ADAPTIVE_BETA",
    ("grpo", "target_kl"): "GRPO_TARGET_KL",
    ("grpo", "kl_stop"): "GRPO_KL_STOP",
    ("evaluation", "samples"): "EVAL_SAMPLES",
}


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    for section in ("pretrain", "sft", "grpo", "evaluation"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"missing configuration section: {section}")
    for section, key in ENV_MAP:
        if key not in config[section]:
            raise ValueError(f"missing configuration value: {section}.{key}")
    if config["grpo"]["target_kl"] <= 0:
        raise ValueError("grpo.target_kl must be positive")
    if config["evaluation"]["samples"] <= 0:
        raise ValueError("evaluation.samples must be positive")
    conditions = config["evaluation"].get("conditions")
    allowed = {"correct", "black", "shuffled"}
    if not conditions or not set(conditions).issubset(allowed):
        raise ValueError("evaluation.conditions must use correct/black/shuffled")
    return config


def config_environment(config, base=None):
    environment = dict(base or os.environ)
    for path, variable in ENV_MAP.items():
        value = config[path[0]][path[1]]
        if isinstance(value, bool):
            value = int(value)
        environment[variable] = str(value)
    environment["PIPELINE_SEED"] = str(config.get("seed", 42))
    environment["EVAL_CONDITIONS"] = " ".join(config["evaluation"]["conditions"])
    return environment


def main():
    parser = argparse.ArgumentParser(description="Run the reproducible MiniMind-V experiment")
    parser.add_argument("--config", default="configs/full_pipeline_5090.json")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    environment = config_environment(config)
    command = ["bash", str(root / "scripts/run_full_training_pipeline.sh")]
    print(json.dumps({"experiment": config.get("name"), "command": command}, ensure_ascii=False))
    if not args.dry_run:
        subprocess.run(command, cwd=root, env=environment, check=True)


if __name__ == "__main__":
    main()
