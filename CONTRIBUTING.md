# Contributing

1. Keep datasets, credentials and checkpoints out of version control.
2. Add or update a JSON experiment config for behavior-changing training changes.
3. Run `make test` and `make lint-config` before opening a pull request.
4. Report the seed, checkpoint, dataset split, decoding settings and evaluator for every metric.
5. Preserve negative results and regression evidence; do not replace held-out metrics with training reward.
