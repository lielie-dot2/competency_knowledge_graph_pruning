
import numpy as np
import embeddings_utils
import numpy as np
from typing import Any, Dict, List

def compute_distance_euclidean(entity_id:str, embedding_hashmap:Dict[Any, List[float]], reference_emb:np.ndarray)->float:
    """Returns the euclidean distance between two entities. """
    emb = embeddings_utils.get_hashmap_content(entity_id, embedding_hashmap)
    if emb is None:
        return float("inf")
    return np.linalg.norm(np.array(emb) - np.array(reference_emb))


def compute_distance_mean(from_id:str, embedding_hashmap:Dict[Any, List[float]], from_emb_ref:np.ndarray, to_id:str, to_emb_ref:np.ndarray)->float:
    """Returns the mean  distance between two  pairs of entities. """
    from_emb = embeddings_utils.get_hashmap_content(from_id, embedding_hashmap)
    to_emb = embeddings_utils.get_hashmap_content(to_id, embedding_hashmap)

    if from_emb is None or to_emb is None:
        return float("inf")

    dist_from = np.linalg.norm(np.array(from_emb) - np.array(from_emb_ref))
    dist_to = np.linalg.norm(np.array(to_emb) - np.array(to_emb_ref))

    return (dist_from + dist_to)/2


def compute_distance_barycentre_euclidean(from_id:str, to_id:str, emb_from_ref:np.ndarray, emb_to_ref:np.ndarray, embedding_hashmap:Dict[Any, List[float]])->float:
    """Returns the euclidean distance of the barycenter of two pairs of entities. """
    from_emb = embeddings_utils.get_hashmap_content(from_id, embedding_hashmap)
    to_emb = embeddings_utils.get_hashmap_content(to_id, embedding_hashmap)

    if from_emb is None or to_emb is None:
        return float("inf")

    known_bary = (np.array(emb_from_ref) + np.array(emb_to_ref)) / 2
    unknown_bary = (np.array(from_emb) + np.array(to_emb)) / 2

    return np.linalg.norm(known_bary - unknown_bary)


def compute_distance_cosine(entity_id:str, embedding_hashmap:Dict[Any, List[float]], reference_emb:str)->float:
    """Returns the cosine distance between two entities. """
    emb = embeddings_utils.get_hashmap_content(entity_id, embedding_hashmap)
    if emb is None:
        return float("inf")

    dot = np.dot(reference_emb, emb)
    norm_product = np.linalg.norm(reference_emb) * np.linalg.norm(emb)
    if norm_product == 0:
        return float("inf")

    return 1 - (dot / norm_product)


def transpose(embedding:np.ndarray)->np.ndarray:
    """Transpose the embedding vector of an entity. """
    return np.transpose(np.array([embedding]), (1, 0))
