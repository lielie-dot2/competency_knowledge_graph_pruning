import argparse
import logging
import pickle
import os
import lmdb
import numpy as np
import pandas as pd
import tqdm
import embeddings_utils
def main():
    parser = argparse.ArgumentParser(prog="generate_sequenced_decisions", description="Generate simple sequenced decisions for training and inference")
    parser.add_argument("--decisions1", dest="decisions1_path", help="CSV decision file for DS1", required=True)
    parser.add_argument("--decisions2", dest="decisions2_path", help="CSV decision file for DS2", required=True)
    parser.add_argument("--decisions3", dest="decisions3_path", help="CSV decision file for DS3", required=True)
    parser.add_argument("--embeddings", dest="embeddings_lmdb_path", help="Folder containing the LMDB embeddings hashmap", required=True)
    parser.add_argument("--output", dest="output_pickle_path", help="Output pickle hashmap", required=True)
    args = parser.parse_args()

    # Logging setup
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Load embeddings (QID URL -> embedding)
    embeddings_env = lmdb.open(args.embeddings_lmdb_path, readonly=True, readahead=False)
    embeddings_txn = embeddings_env.begin()

    # Load decision files and combine them
    df_decisions1 = pd.read_csv(args.decisions1_path).dropna()
    df_decisions2 = pd.read_csv(args.decisions2_path).dropna()
    df_decisions3 = pd.read_csv(args.decisions3_path).dropna()
    df_decisions = pd.concat([df_decisions1, df_decisions2, df_decisions3], axis=0)
    source_qids = set(df_decisions["from"])
    all_sequences = {}
    for source_qid in tqdm.tqdm(source_qids):
        source_embedding = embeddings_utils.get_hashmap_content(source_qid, embeddings_txn)
        if source_embedding is not None:
            target_sequences = {}
            target_qids = set(df_decisions[df_decisions["from"] == source_qid]["QID"])
            for target_qid in target_qids:
                target_embedding = embeddings_utils.get_hashmap_content(target_qid, embeddings_txn)
                if target_embedding is not None:
                    # Concatenate embeddings of source and target
                    sequence = [np.transpose(np.array([source_embedding]), (1, 0)),np.transpose(np.array([target_embedding]))]
                    target_sequences[target_qid] = sequence
            all_sequences[source_qid] = target_sequences
    # Merge with existing sequences if exists
    if os.path.exists(args.output_pickle_path):
        with open(args.output_pickle_path, "rb") as file:
            existing_sequences = pickle.load(file)
        existing_sequences.update(all_sequences)
        print("Merging with existing sequences...")
        all_sequences = existing_sequences
    # Save final sequences to pickle
    with open(args.output_pickle_path, "wb") as file:
        pickle.dump(all_sequences, file)


if __name__ == '__main__':
    main()

