"""Reusable preprocessing helpers for capstone notebooks."""
def to_dense(array):
    return array.toarray() if hasattr(array, "toarray") else array
