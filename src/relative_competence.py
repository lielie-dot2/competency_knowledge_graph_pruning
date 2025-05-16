import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import log_loss
import argparse
import yaml
import logging
import embeddings_utils


def compute_relative_competence(model_path : str, sequence_file : str, decisions_file : str ,output_file :str,dim : int ,sequence_length :int , padding: str) -> pd.DataFrame: 
    """ Computes the log-loss competency score the left-pairs of each trained model dataset. """
    logger = logging.getLogger(__name__),
    analogy_classifier = tf.keras.models.load_model(model_path)
    sequenced_decisions = pickle.load(open(sequence_file, "rb"))
    all_decisions = pd.read_csv(decisions_file).dropna()
    true_decisions = all_decisions[all_decisions["target"] == 1]
    true_decisions_array = true_decisions.to_numpy()
    all_decisions_array = all_decisions.to_numpy()
    competences = []
    for true_decision in tqdm(true_decisions_array, desc=f"Computing competence for {os.path.basename(decisions_file)}"):
        qid_C, _, class_D = true_decision[:3]
        vec_CD = None
        if qid_C in sequenced_decisions and class_D in sequenced_decisions[qid_C]:
            seq_CD = sequenced_decisions[qid_C][class_D]
            vec_CD = embeddings_utils.pad_sequence(seq_CD, sequence_length, dim, padding)
        analogies = []
        expected_targets = []
        if vec_CD is None:
            logger.warning(f"Sequenced decision not found for QID {qid_C} class {class_D}")
            competences.append(1000)
            continue
        for other_decision in all_decisions_array:
            if np.array_equal(true_decision[:3], other_decision[:3]):
                continue
            qid_A, _, class_B = other_decision[:3]
            vec_AB = None
            if qid_A in sequenced_decisions and class_B in sequenced_decisions[qid_A]:
                seq_AB = sequenced_decisions[qid_A][class_B]
                vec_AB = embeddings_utils.pad_sequence(seq_AB, sequence_length, dim, padding)
            if vec_AB is not None:
                embeddings_utils.add_evaluation_sequence_analogy(analogies, vec_CD, vec_AB, properties=".")
                expected_targets.append(other_decision[5])
        if analogies:
            analogies = np.transpose(analogies, (0, 2, 1, 3))
            predictions = analogy_classifier.predict(analogies)
            predicted_probs = predictions.squeeze()
            expected_targets_array = np.array(expected_targets)
            competence = log_loss(expected_targets_array, predicted_probs, labels=[0, 1])
        else:
            logger.info("No analogies available for this decision.")
            competence = 1000
        competences.append(competence)
    true_decisions["competence"] = competences
    true_decisions = true_decisions.sort_values("competence", ascending=False)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    true_decisions.to_csv(output_file, index=False)
    logger.info(f"Saved competence scores to {output_file}")
    return true_decisions


def main():
    """ Main function to load configuration and compute relative competence for all models. """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Compute relative competence")
    parser.add_argument("--config",type=str,default="./configuration/config.yml",help="Path to the configuration YAML file",)
    args = parser.parse_args()
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    sequence_length = config.get("sequence_length", 4)
    padding = config.get("padding", "between")
    dim = config.get("dim", 300)
    decisions_file = config.get("decisions_file")
    models = config.get("models", [])
    for model_path in models:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        dataset_id = ''.join(filter(str.isdigit, model_name))[-1]
        decision_file = f"./data/dataset{dataset_id}_gold_decisions_train.csv"
        output_file = f"./data/competence_scores/{model_name}_competence_logloss.csv"
        logger.info(f"Processing model: {model_path}")
        compute_relative_competence(model_path=model_path,sequence_file=decisions_file,decisions_file=decision_file,output_file=output_file,dim=dim,sequence_length=sequence_length,padding=padding)


if __name__ == "__main__":
    main()
