import pytest

pytest.importorskip("torch")

from eval_vlm_qa import normalize_answer, summarize_pope, vqa_consensus


def test_vqa_answer_normalization():
    assert normalize_answer("The two, cats!") == "2 cats"


def test_vqa_consensus_uses_annotator_agreement():
    answers = ["two"] * 6 + ["three"] * 4
    assert vqa_consensus("two", answers) > vqa_consensus("three", answers)


def test_pope_metrics():
    rows = [
        {"prediction_label": "yes", "label": "yes"},
        {"prediction_label": "yes", "label": "no"},
        {"prediction_label": "no", "label": "no"},
        {"prediction_label": "no", "label": "yes"},
    ]
    result = summarize_pope(rows)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
