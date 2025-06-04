def score(prediction, gold_label):
    """Return the score for a single prediction and gold label."""
    return 1.0 if prediction == gold_label else 0.0
