from pathlib import Path

import pytest

from scripts.summarize_experiment import training_summary


def test_training_summary_uses_last_completion(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "Training complete, elapsed: 1.000h, peak_vram: 2.000GB\n"
        "Training complete, elapsed: 3.500h, peak_vram: 4.250GB\n",
        encoding="utf-8",
    )
    assert training_summary(log) == {"elapsed_hours": 3.5, "peak_vram_gb": 4.25}


def test_training_summary_requires_completion(tmp_path):
    log = Path(tmp_path) / "train.log"
    log.write_text("still running", encoding="utf-8")
    with pytest.raises(ValueError, match="completion line"):
        training_summary(log)
