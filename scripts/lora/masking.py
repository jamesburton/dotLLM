"""Completion-only loss masking: supervise only the assistant completion tokens."""

def build_labels(prompt_ids: list[int], completion_ids: list[int]) -> tuple[list[int], list[int]]:
    input_ids = list(prompt_ids) + list(completion_ids)
    labels = [-100] * len(prompt_ids) + list(completion_ids)
    return input_ids, labels
