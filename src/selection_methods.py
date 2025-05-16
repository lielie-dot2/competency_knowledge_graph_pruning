import random
import numpy as np
import embeddings_utils
from distance_calculation import (
    compute_distance_euclidean,
    compute_distance_mean,
    compute_distance_mean_b,
    compute_distance_barycentre_euclidean,
    compute_distance_barycentre_euclidean_b,
)
import pandas as pd
from typing import List, Optional, Any


def get_distance_target(keeping_decisions: pd.DataFrame, start_entity: str, reached_entity: str, nb_keeping_in_test: int, embedding_hashmap: Any) -> Optional[List[List[Any]]]:
    """ Returns the k closest left-pairs based on distance to the target embedding. """
    known_reached_emb = embeddings_utils.get_hashmap_content(reached_entity, embedding_hashmap)
    if known_reached_emb is None:
        return None
    filtered = keeping_decisions.copy()
    filtered["distance"] = filtered["QID"].apply(lambda x: compute_distance_euclidean(x, embedding_hashmap, known_reached_emb))
    filtered = filtered.sort_values(by="distance")
    return filtered.head(nb_keeping_in_test).values.tolist()


def get_distance_source(keeping_decisions: pd.DataFrame, start_entity: str, reached_entity: str, nb_keeping_in_test: int, embedding_hashmap: Any) -> Optional[List[List[Any]]]:
    """ Returns the k closest left-pairs based on distance to the source embedding. """
    known_source_emb = embeddings_utils.get_hashmap_content(start_entity, embedding_hashmap)
    if known_source_emb is None:
        return None
    filtered = keeping_decisions.copy()
    filtered["distance"] = filtered["from"].apply(lambda x: compute_distance_euclidean(x, embedding_hashmap, known_source_emb))
    filtered = filtered.sort_values(by="distance")
    return filtered.head(nb_keeping_in_test).values.tolist()


def get_distance_pairs(keeping_decisions: pd.DataFrame, start_entity: str, reached_entity: str, nb_keeping_in_test: int, embedding_hashmap: Any) -> List[List[Any]]:
    """ Returns k closest left-pairs based on the mean distance between pairs of entities. """
    start_emb = embeddings_utils.get_hashmap_content(start_entity, embedding_hashmap)
    reached_emb = embeddings_utils.get_hashmap_content(reached_entity, embedding_hashmap)
    if start_emb is None or reached_emb is None:
        return []
    filtered = keeping_decisions.copy()
    filtered["distance"] = filtered.apply(lambda row: compute_distance_mean(row["from"], embedding_hashmap, start_emb, row["QID"], reached_emb),axis=1)
    filtered = filtered.sort_values(by="distance")
    return filtered.head(nb_keeping_in_test).values.tolist()


def get_distance_pairs_b(keeping_decisions: pd.DataFrame, start_entity: str, reached_entity: str, nb_keeping_in_test: int, embedding_hashmap: Any) -> List[List[Any]]:
    """" Returns k closest left-pairs based on the mean distance between heads and tails of pairs of entities. """
    start_emb = embeddings_utils.get_hashmap_content(start_entity, embedding_hashmap)
    reached_emb = embeddings_utils.get_hashmap_content(reached_entity, embedding_hashmap)
    if start_emb is None or reached_emb is None:
        return []
    filtered = keeping_decisions.copy()
    filtered["distance"] = filtered.apply(lambda row: compute_distance_mean_b(row["from"], embedding_hashmap, start_emb, row["QID"], reached_emb),axis=1)
    filtered = filtered.sort_values(by="distance")
    return filtered.head(nb_keeping_in_test).values.tolist()


def get_distance_barycentre(keeping_decisions: pd.DataFrame, start_entity: str, reached_entity: str, nb_keeping_in_test: int, embedding_hashmap: Any) -> List[List[Any]]:
    """ Returns top-k closest pairs based on barycentre distance between pairs of entities. """
    start_emb = embeddings_utils.get_hashmap_content(start_entity, embedding_hashmap)
    reached_emb = embeddings_utils.get_hashmap_content(reached_entity, embedding_hashmap)
    if start_emb is None or reached_emb is None:
        return []
    filtered = keeping_decisions.copy()
    filtered["distance"] = filtered.apply(lambda row: compute_distance_barycentre_euclidean(row["from"], row["QID"], start_emb, reached_emb, embedding_hashmap), axis=1)
    filtered = filtered[filtered["distance"] > 0].sort_values(by="distance")
    return filtered.head(nb_keeping_in_test).values.tolist()


def get_distance_barycentre_b(keeping_decisions: pd.DataFrame, start_entity: str, reached_entity: str, nb_keeping_in_test: int, embedding_hashmap: Any) -> List[List[Any]]:
    """ Returns top-k closest pairs based on barycentre distance between hheads and tails of pairs of entities. """
    start_emb = embeddings_utils.get_hashmap_content(start_entity, embedding_hashmap)
    reached_emb = embeddings_utils.get_hashmap_content(reached_entity, embedding_hashmap)
    if start_emb is None or reached_emb is None:
        return []
    filtered = keeping_decisions.copy()
    filtered["distance"] = filtered.apply(lambda row: compute_distance_barycentre_euclidean_b(row["from"], row["QID"], start_emb, reached_emb, embedding_hashmap),axis=1)
    filtered = filtered[filtered["distance"] > 0].sort_values(by="distance")
    return filtered.head(nb_keeping_in_test).values.tolist()


def get_random_pairs(keeping_decisions: pd.DataFrame, start_entity: str, reached_entity: str, nb_keeping_in_test: int, embedding_hashmap: Any = None) -> List[List[Any]]:
    """Returns a random selection of left-pairs. """
    if len(keeping_decisions) == 0:
        return []
    kept_start = random.sample(keeping_decisions["from"].tolist(), min(nb_keeping_in_test, len(keeping_decisions)))
    return keeping_decisions[keeping_decisions["from"].isin(kept_start)].values.tolist()


def get_pairs_relative_competence(keeping_decisions: pd.DataFrame, start_entity: str, reached_entity: str,nb_keeping_in_test: int, embedding_hashmap: Any = None) -> List[List[Any]]:
    """ Returns the k closest left-pairs based on relative competency score (lower log loss -> higher competency)"""
    return keeping_decisions.sort_values(by="competence").head(nb_keeping_in_test).values.tolist()
