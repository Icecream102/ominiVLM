from evaluation.caption_metrics import compute_caption_metrics, jaccard_distance, tokenize


def test_tokenize_mixed_text():
    assert tokenize("A dog, 两只猫!") == ["a", "dog", "两", "只", "猫"]


def test_identical_predictions_score_higher_than_unrelated():
    references = [["a dog runs on grass"], ["two cats sit together"]]
    identical = compute_caption_metrics([refs[0] for refs in references], references)
    unrelated = compute_caption_metrics(["red bus", "blue sky"], references)
    assert identical["BLEU-4"] > unrelated["BLEU-4"]
    assert identical["ROUGE-L"] > unrelated["ROUGE-L"]
    assert identical["CIDEr"] > unrelated["CIDEr"]


def test_jaccard_distance():
    assert jaccard_distance("a black cat", "a black cat") == 0.0
    assert jaccard_distance("a black cat", "two white dogs") == 1.0

