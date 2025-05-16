import pickle
import numpy as np
import json
from typing import Any, Dict, List, Optional


def get_hashmap_content(key: str, hashmap: Any) -> Optional[Any]:
    """Retrieve and deserialize an object from a hashmap by key."""
    obj = hashmap.get(key.encode("ascii"))
    return pickle.loads(obj) if obj else None


def pad_sequence(seq: List[np.ndarray], max_length: int, dim: int, pad_mode: str = "after") -> List[np.ndarray]:
    """Pad or truncate a sequence of vectors to a fixed length."""
    pad = max_length - len(seq)
    zero_vector = np.zeros((dim, 1))

    if pad >= 0:
        if pad_mode == "after":
            return seq + [zero_vector] * pad
        elif pad_mode == "before":
            return [zero_vector] * pad + seq
        else:  # between
            return [seq[0]] + [zero_vector] * pad + seq[1:]
    else:
        return [seq[0]] + seq[-pad+1:]


def add_evaluation_sequence_analogy(test_set: List[List[np.ndarray]], vec_AB: List[np.ndarray], vec_CD: List[np.ndarray], properties: str) -> None:
    """Add analogy pairs for evaluation, with optional symmetry and inner-symmetry."""
    test_set.append(vec_AB + vec_CD)

    if "symmetry" in properties:
        test_set.append(vec_CD + vec_AB)

    if "inner-symmetry" in properties:
        vec_AB.reverse()
        vec_CD.reverse()
        test_set.append(vec_AB + vec_CD)
        if "symmetry" in properties:
            test_set.append(vec_CD + vec_AB)
        vec_AB.reverse()
        vec_CD.reverse()


def add_sequence_analogy(mode: str, vec_AB: List[np.ndarray], vec_CD: List[np.ndarray], properties: str,
                         valid_analogies_pattern: List[str], analogy_pattern: str,
                         features: List[List[np.ndarray]], labels: List[int]) -> None:
    """Add training or test analogies with corresponding labels based on analogy pattern and properties."""
    if analogy_pattern in ["kk", "pp"]:
        features.append(vec_AB + vec_CD)
        add_label(analogy_pattern, labels, valid_analogies_pattern)

        if "symmetry" in properties:
            features.append(vec_CD + vec_AB)
            add_label(analogy_pattern, labels, valid_analogies_pattern)

        if analogy_pattern in valid_analogies_pattern and mode == "train" and "reflexivity" in properties:
            features.extend([vec_AB + vec_AB, vec_CD + vec_CD])
            add_label(analogy_pattern, labels, valid_analogies_pattern)
            add_label(analogy_pattern, labels, valid_analogies_pattern)

        if "inner-symmetry" in properties:
            vec_AB.reverse()
            vec_CD.reverse()
            features.append(vec_AB + vec_CD)
            add_label(analogy_pattern, labels, valid_analogies_pattern)
            if "symmetry" in properties:
                features.append(vec_CD + vec_AB)
                add_label(analogy_pattern, labels, valid_analogies_pattern)
            if analogy_pattern in valid_analogies_pattern and mode == "train" and "relexivity" in properties:
                features.extend([vec_AB + vec_AB, vec_CD + vec_CD])
                add_label(analogy_pattern, labels, valid_analogies_pattern)
                add_label(analogy_pattern, labels, valid_analogies_pattern)
            vec_AB.reverse()
            vec_CD.reverse()

    if analogy_pattern in ["kp", "pk"]:
        features.append(vec_AB + vec_CD)
        add_label(analogy_pattern, labels, valid_analogies_pattern)

        if "symmetry" in properties:
            features.append(vec_CD + vec_AB)
            add_label(analogy_pattern, labels, valid_analogies_pattern)

        if "inner-symmetry" in properties:
            vec_AB.reverse()
            vec_CD.reverse()
            features.append(vec_AB + vec_CD)
            add_label(analogy_pattern, labels, valid_analogies_pattern)
            if "symmetry" in properties:
                features.append(vec_CD + vec_AB)
                add_label(analogy_pattern, labels, valid_analogies_pattern)
            vec_AB.reverse()
            vec_CD.reverse()


def add_label(analogy_pattern: str, labels: List[int], valid: List[str]) -> None:
    """Append 1 if analogy is valid, else 0."""
    if analogy_pattern in valid:
        labels.append(1)
    else:
        labels.append(0)
