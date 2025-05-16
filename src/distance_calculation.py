import numpy as np
import pandas as pd
import embeddings_utils

def compute_distance_euclidean(entity_id, embedding_hashmap, reference_emb) -> float:
    """ Return the Euclidean distance between the embedding of two entities. """
    emb = embeddings_utils.get_hashmap_content(entity_id, embedding_hashmap)
    if emb is None:
        return float("inf")
    return np.linalg.norm(np.array(emb) - np.array(reference_emb))

def compute_distance_mean(from_id, embedding_hashmap, from_emb_ref, to_id, to_emb_ref) -> float:
    """ Returns the average of the distance between heads and tails entities of the pairs. """
    from_emb = embeddings_utils.get_hashmap_content(from_id, embedding_hashmap)
    to_emb = embeddings_utils.get_hashmap_content(to_id, embedding_hashmap)
    if from_emb is None or to_emb is None:
        return float("inf")
    dist_from = np.linalg.norm(np.array(from_emb) - np.array(from_emb_ref))
    dist_to = np.linalg.norm(np.array(to_emb) - np.array(to_emb_ref))
    return (dist_from + dist_to) /2

def compute_distance_mean_b(from_id, embedding_hashmap, from_emb_ref, to_id, to_emb_ref) -> float:
    """ Returns the average of the distance between entities of each pair. """
    from_emb = embeddings_utils.get_hashmap_content(from_id, embedding_hashmap)
    to_emb = embeddings_utils.get_hashmap_content(to_id, embedding_hashmap)
    if from_emb is None or to_emb is None:
        return float("inf")
    dist_start = np.linalg.norm(np.array(from_emb) - np.array(to_emb))
    dist_reached = np.linalg.norm(np.array(from_emb_ref) - np.array(to_emb_ref))
    return (dist_start + dist_reached) / 2

def compute_distance_barycentre_euclidean(from_id, to_id, emb_from_ref, emb_to_ref, embedding_hashmap) -> float:
    """ Returns the Euclidean distance between the barycentres of heads and tails of the two pairs."""
    from_emb = embeddings_utils.get_hashmap_content(from_id, embedding_hashmap)
    to_emb = embeddings_utils.get_hashmap_content(to_id, embedding_hashmap)
    if from_emb is None or to_emb is None:
        return float("inf")
    known_bary = (np.array(emb_from_ref) + np.array(from_emb)) / 2
    unknown_bary = (np.array(emb_to_ref) + np.array(to_emb)) / 2
    return np.linalg.norm(known_bary - unknown_bary)

def compute_distance_barycentre_euclidean_b(from_id, to_id, emb_from_ref, emb_to_ref, embedding_hashmap) -> float:
    """ Returns the Euclidean distance of the barycentres of each pair. """
    from_emb = embeddings_utils.get_hashmap_content(from_id, embedding_hashmap)
    to_emb = embeddings_utils.get_hashmap_content(to_id, embedding_hashmap)
    if from_emb is None or to_emb is None:
        return float("inf")
    known_bary = (np.array(emb_from_ref) + np.array(emb_to_ref)) / 2
    unknown_bary = (np.array(from_emb) + np.array(to_emb)) / 2
    return np.linalg.norm(known_bary - unknown_bary)

def transpose(embedding) -> np.ndarray:
    """ Returns the transposed version of the input embedding. """
    return np.transpose(np.array([embedding]), (1, 0))
