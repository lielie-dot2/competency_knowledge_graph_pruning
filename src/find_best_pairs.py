import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import random
import tensorflow as tf
import lmdb
import pickle
import pandas as pd
import embeddings_utils
from typing import Callable, Optional
import argparse
import yaml
import logging
from sklearn.metrics import f1_score
from selection_methods import (get_distance_barycentre,get_distance_source,get_distance_pairs,get_distance_target,get_pairs_relative_competence,get_random_pairs)

logging.basicConfig(level=logging.INFO)

def load_config(config_file:str):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def get_method_function(method_name: str) -> Optional[Callable] :
    method_mapping = {
        "M1": get_distance_pairs,
        "M2": get_distance_source,
        "M3": get_distance_target,
        "M4": get_distance_barycentre,
        "M5": get_pairs_relative_competence,
        "Random": get_random_pairs
    }
    return method_mapping.get(method_name, None)

def grid_search_nb_keeping(model_path, embeddings_path, decisions_file, train_file, test_file, min_val, max_val, step, dim, method, threshold,sequence_length,padding):
    """Find the optimal weighted F-measure and the optimal number of pairs for each experiment. """
    f1_scores = []
    trials = np.arange(min_val, max_val + 1, step)
    best_f1 = -1
    best_nb_keeping = None
    for nb_keeping in trials:
        if method == "Random":
            scores = [evaluate_sequenced_analogy_model(model_path, embeddings_path, decisions_file, train_file, test_file,sequence_length, padding, nb_keeping_in_test=nb_keeping, dim=dim,method=method, threshold=threshold) for _ in range(150)]
            f1 = np.mean(scores)
            print(np.std(scores))
        else:
            f1 = evaluate_sequenced_analogy_model(model_path, embeddings_path, decisions_file, train_file, test_file,sequence_length, padding, nb_keeping_in_test=nb_keeping, dim=dim,method=method, threshold=threshold)
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_nb_keeping = nb_keeping
    return best_nb_keeping, best_f1

def evaluate_sequenced_analogy_model(model_path : str, embeddings_path : str, decisions_file : str, train_file :str , test_file : str,sequence_length : int, padding :str, nb_keeping_in_test: int, dim: int, method :str, threshold: float):
    sequenced_analogy_classifier = tf.keras.models.load_model(model_path)
    embeddings = lmdb.open(embeddings_path, readonly=True, readahead=False)
    embedding_hashmap = embeddings.begin()
    sequenced_decisions = pickle.load(open(decisions_file, "rb"))
    keeping_decisions_train = pd.read_csv(train_file).dropna()
    decisions_test = pd.read_csv(test_file).dropna()
    keeping_decisions_train = keeping_decisions_train[keeping_decisions_train["target"] == 1]
    decisions_test_array = decisions_test.to_numpy()
    keeping_analogies = []
    expected_target = []
    method_function = get_method_function(method)
    if method_function is None:
        raise ValueError(f"Unknown method: {method}")
    for decision in decisions_test_array:
        starting_qid_C = decision[0]
        reached_class_D = decision[2]
        if starting_qid_C in sequenced_decisions and reached_class_D in sequenced_decisions[starting_qid_C]:
            vec_kept_CD = embeddings_utils.pad_sequence(sequenced_decisions[starting_qid_C][reached_class_D], sequence_length, dim, padding)
            known_keeping_pair = method_function(keeping_decisions_train, decision[0], decision[2], nb_keeping_in_test, embedding_hashmap)
            if known_keeping_pair is not None:
                for keeping_pair in known_keeping_pair:
                    starting_qid_A = keeping_pair[0]
                    reached_class_B = keeping_pair[2]
                    if starting_qid_A in sequenced_decisions and reached_class_B in sequenced_decisions[starting_qid_A]:
                        vec_kept_AB = embeddings_utils.pad_sequence(sequenced_decisions[starting_qid_A][reached_class_B], sequence_length, dim, padding)
                        embeddings_utils.add_evaluation_sequence_analogy(keeping_analogies, vec_kept_AB, vec_kept_CD, properties=".")
                        expected_target.append(decision[5])
    keeping_analogies = np.transpose(keeping_analogies, (0, 2, 1, 3))
    y_keeping_pred = sequenced_analogy_classifier.predict(keeping_analogies, verbose=0)
    predicted_decision = np.squeeze(y_keeping_pred)
    max_batches = len(predicted_decision) // nb_keeping_in_test
    true_class = []
    predicted_class = []
    for k in range(max_batches):
        true_class.append(int(np.mean(expected_target[k * nb_keeping_in_test: (k + 1) * nb_keeping_in_test])))
        votes = [1 if i > threshold else 0 for i in predicted_decision[k * nb_keeping_in_test: (k + 1) * nb_keeping_in_test]]
        vote_for_keep = sum(votes)
        predicted_class.append(1 if vote_for_keep > (nb_keeping_in_test * 0.5) else 0)
    embeddings.close()
    return f1_score(predicted_class, true_class, average="weighted")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    config = load_config(args.config)
    threshold = config["threshold"]
    sequence_length = config["sequence_length"]
    padding = config["padding"]
    dim = config["dim"]
    embeddings_path = config["embeddings_path"]
    decisions_file = config["decisions_file"]
    models = config["models"]
    train_files = config["train_files"]
    test_files = config["test_files"]
    methods = config["methods"]
    output_file = config["output_file"]
    grid_min = config["grid_search"]["min_val"]
    grid_max = config["grid_search"]["max_val"]
    step = config["grid_search"]["step"]
    results = []
    for model_path in models:
        for train_file in train_files:
            for test_file in test_files:
                print("Hi")
                logging.info(f"Running grid search for model: {model_path}, train: {train_file}, test: {test_file}")
                for method in methods:
                    best_nb, best_f1 = grid_search_nb_keeping(model_path=model_path,embeddings_path=embeddings_path,decisions_file=decisions_file,train_file=train_file,test_file=test_file,min_val=grid_min,max_val=grid_max,step=step,dim=dim,method=method,threshold=threshold,sequence_length=sequence_length,padding=padding)
                    results.append({
                        "model": os.path.basename(model_path),
                        "train_file": os.path.basename(train_file),
                        "test_file": os.path.basename(test_file),
                        "method": method,
                        "optimal_nb_keeping": best_nb,
                        "optimal_f1": best_f1
                    })
    with open(output_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["model", "train_file", "test_file", "method", "optimal_nb_keeping", "optimal_f1"])
        writer.writeheader()
        writer.writerows(results)
    logging.info(f"\nAll results were saved to: {output_file}")
if __name__ == "__main__":
    main()
