import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from masking import build_labels

def test_prompt_is_masked_completion_is_supervised():
    prompt = [10, 11, 12]
    completion = [20, 21]
    ids, labels = build_labels(prompt, completion)
    assert ids == [10, 11, 12, 20, 21]
    assert labels == [-100, -100, -100, 20, 21]

def test_empty_completion_supervises_nothing():
    ids, labels = build_labels([1, 2], [])
    assert ids == [1, 2]
    assert labels == [-100, -100]
