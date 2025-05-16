"""
Copyright (C) 2023 Orange
Authors: Lucas Jarnac, Miguel Couceiro, and Pierre Monnin

This software is distributed under the terms and conditions of the 'MIT'
license which can be found in the file 'LICENSE.txt' in this package distribution 
or at 'https://opensource.org/license/mit/'.
"""
import logging
import tqdm
import random
import pickle
import argparse
import os
import lmdb
import numpy as np
import pandas
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import f1_score
import tensorflow as tf
import optuna
import embeddings_utils


def sequenced_analogy_model(seq_len, nb_filters1, nb_filters2, dropout, emb_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=nb_filters1, 
                               kernel_size=(1, seq_len), 
                               strides=(1, seq_len), 
                               activation="relu", 
                               input_shape=(emb_size, seq_len*2, 1), 
                               kernel_initializer="he_normal"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Conv2D(filters=nb_filters2, 
                               kernel_size=(2, 2), 
                               strides=(2, 2), 
                               activation="relu", 
                               kernel_initializer="he_normal"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    return model

    

def transpose(embedding):
    return numpy.transpose(numpy.array([embedding]), (1, 0))


def get_analogies(train_features, 
                  train_labels, 
                  keeping_train, 
                  pruning_train, 
                  nb_training_analogies_per_decision,
                  sequenced_decisions, 
                  sequence_length, 
                  padding,
                  properties,
                  valid_analogies_pattern,
                  invalid_analogies_pattern,
                  embedding_hashmap,
                  knn):

    keeping_decisions = keeping_train.values.tolist()
    pruning_decisions = pruning_train.values.tolist()
    keeping_to_shuffle = keeping_decisions.copy()
    pruning_to_shuffle = pruning_decisions.copy()

    # Keeping decisions :: Keeping decisions
    for i in range(len(keeping_decisions)):
        starting_qid_A = keeping_decisions[i][0]
        reached_class_B = keeping_decisions[i][1]
        if starting_qid_A in sequenced_decisions and reached_class_B in sequenced_decisions[starting_qid_A]:
            vec_kept_AB = embeddings_utils.pad_sequence(sequenced_decisions[starting_qid_A][reached_class_B], sequence_length, 300,padding)
            if "kk" in valid_analogies_pattern + invalid_analogies_pattern:
               
                nb_used_analogies = 0
                training_analogy_index = 0

                keeping = []
                if "train" not in knn:
                
                    random.shuffle(keeping_to_shuffle)
                    keeping = keeping_to_shuffle
                else:
                    # K nearest neighbors
                    keeping = embeddings_utils.get_hashmap_distances(keeping_decisions[i][0],
                                                          keeping_train,
                                                          embedding_hashmap)
                
                while nb_used_analogies < nb_training_analogies_per_decision:
                    if training_analogy_index == len(keeping):
                        break
                    starting_qid_C = keeping[training_analogy_index][0]
                    reached_class_D = keeping[training_analogy_index][1]
                    if starting_qid_C in sequenced_decisions and reached_class_D in sequenced_decisions[starting_qid_C] and \
                        keeping[training_analogy_index][0] != keeping_decisions[i][0]:
                        vec_kept_CD = embeddings_utils.pad_sequence(sequenced_decisions[starting_qid_C][reached_class_D], sequence_length,300, padding)
                        embeddings_utils.add_sequence_analogy("train", vec_kept_CD, vec_kept_AB, properties, valid_analogies_pattern, "kk", train_features, train_labels)
                        
                        
                        nb_used_analogies += 1
                    training_analogy_index += 1
              
           
            
    # Pruning decisions :: Pruning decisions
    for i in range(len(pruning_decisions)):
        
        starting_qid_A = pruning_decisions[i][0]
        reached_class_B = pruning_decisions[i][1]
        if starting_qid_A in sequenced_decisions and reached_class_B in sequenced_decisions[starting_qid_A]:
            vec_pruned_AB = embeddings_utils.pad_sequence(sequenced_decisions[starting_qid_A][reached_class_B], sequence_length,300, padding)
            if "kp" in valid_analogies_pattern + invalid_analogies_pattern:
                nb_used_analogies = 0
                training_analogy_index = 0
                keeping = []
                if "train" not in knn:
                    random.shuffle(keeping_to_shuffle)
                    keeping = keeping_to_shuffle
                else:
                    # K nearest neighbors
                    keeping = embeddings_utils.get_hashmap_distances(pruning_decisions[i][0],
                                                          keeping_train,
                                                          embedding_hashmap)

                # Invalid analogies
                while nb_used_analogies < nb_training_analogies_per_decision:
                    if training_analogy_index == len(keeping):
                        break
                    starting_qid_C = keeping[training_analogy_index][0]
                    reached_class_D = keeping[training_analogy_index][1]
                    if starting_qid_C in sequenced_decisions and reached_class_D in sequenced_decisions[starting_qid_C] and \
                        keeping[training_analogy_index][0] != pruning_decisions[i][0]:
                        vec_kept_CD = embeddings_utils.pad_sequence(sequenced_decisions[starting_qid_C][reached_class_D], sequence_length,300, padding)
                        # Keeping decisions :: Pruning decisions
                        embeddings_utils.add_sequence_analogy("train", vec_kept_CD, vec_pruned_AB, properties, valid_analogies_pattern, "kp", train_features, train_labels)
                        nb_used_analogies += 1
                    training_analogy_index += 1


def get_testing_analogies(test_features, 
                          test_labels, 
                          keeping_decisions_test, 
                          pruning_decisions_test,
                          keeping_train,
                          pruning_train, 
                          nb_test_analogies, 
                          sequenced_decisions,
                          sequence_length,
                          padding,
                          properties,
                          valid_analogies_pattern,
                          invalid_analogies_pattern,
                          embedding_hashmap,
                          knn):
    
    keeping_decisions = keeping_train.values.tolist()
    pruning_decisions = pruning_train.values.tolist()
    keeping_to_shuffle = keeping_decisions.copy()
    pruning_to_shuffle = pruning_decisions.copy()
    # Keeping decisions :: Keeping decisions
    for i in range(len(keeping_decisions_test)):
        
        starting_qid_A = keeping_decisions_test[i][0]
        reached_class_B = keeping_decisions_test[i][1]
        if starting_qid_A in sequenced_decisions and reached_class_B in sequenced_decisions[starting_qid_A]:
            vec_kept_AB = embeddings_utils.pad_sequence(sequenced_decisions[starting_qid_A][reached_class_B], sequence_length,300, padding)
            if "kk" in valid_analogies_pattern + invalid_analogies_pattern and "kp" in invalid_analogies_pattern:
                nb_used_analogies = 0
                training_analogy_index = 0
                keeping = []
                
                if "test" not in knn:
                  
                    random.shuffle(keeping_to_shuffle)
                    keeping = keeping_to_shuffle
                else:
                    # K nearest neighbors
                    keeping = embeddings_utils.get_hashmap_distances(keeping_decisions_test[i][0],
                                                          keeping_train,
                                                          embedding_hashmap)

                while nb_used_analogies < nb_test_analogies:
                    if training_analogy_index == len(keeping):
                        break
                    starting_qid_C = keeping[training_analogy_index][0]
                    reached_class_D = keeping[training_analogy_index][1]
                    if starting_qid_C in sequenced_decisions and reached_class_D in sequenced_decisions[starting_qid_C]:
                        vec_kept_CD = embeddings_utils.pad_sequence(sequenced_decisions[starting_qid_C][reached_class_D], sequence_length,300, padding)
                        embeddings_utils.add_sequence_analogy("test", vec_kept_CD, vec_kept_AB, properties, valid_analogies_pattern, "kk", test_features, test_labels)
                        nb_used_analogies += 1
                    training_analogy_index += 1
            
    # Pruning decisions :: Pruning decisions
    for i in range(len(pruning_decisions_test)):
       
        starting_qid_A = pruning_decisions_test[i][0]
        reached_class_B = pruning_decisions_test[i][1]
        if starting_qid_A in sequenced_decisions and reached_class_B in sequenced_decisions[starting_qid_A]:
            vec_pruned_AB = embeddings_utils.pad_sequence(sequenced_decisions[starting_qid_A][reached_class_B], sequence_length, 300,padding)
            if "kp" in valid_analogies_pattern + invalid_analogies_pattern:
                nb_used_analogies = 0
                training_analogy_index = 0
             
                keeping = []
                if "test" not in knn:
                    random.shuffle(keeping_to_shuffle)
                    keeping = keeping_to_shuffle
                else:
                    # K nearest neighbors
                    keeping = embeddings_utils.get_hashmap_distances(pruning_decisions_test[i][0],
                                                          keeping_train,
                                                          embedding_hashmap)

                # Invalid analogies
                while nb_used_analogies < nb_test_analogies:
                    if training_analogy_index == len(keeping):
                        break
                    starting_qid_C = keeping[training_analogy_index][0]
                    reached_class_D = keeping[training_analogy_index][1]
                    if starting_qid_C in sequenced_decisions and reached_class_D in sequenced_decisions[starting_qid_C]:
                        vec_kept_CD = embeddings_utils.pad_sequence(sequenced_decisions[starting_qid_C][reached_class_D], sequence_length,300 ,padding)
                        # Keeping decisions :: Pruning decisions
                        embeddings_utils.add_sequence_analogy("test", vec_kept_CD, vec_pruned_AB, properties, valid_analogies_pattern, "kp", test_features, test_labels)
                        nb_used_analogies += 1
                    training_analogy_index += 1

def main():
    SEED = 58 
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    tf.config.experimental.enable_op_determinism()
    sampler = optuna.samplers.TPESampler(seed=SEED)  
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--sequenced-decisions", dest="sequenced_decisions", required=True)
    parser.add_argument("--decisions", dest="decisions", help="CSV decision file", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    logger = logging.getLogger()

    params = {
        "knn": ".",
        "valid_analogies_pattern": "kk", 
        "invalid_analogies_pattern": "kp",
        "nb_training_analogies_per_decision":20,
        "nb_test_analogies": 20, 
        "sequence_length": 4, 
        "padding": "between", 
        "analogical_properties": ["."],
        "nb_filters": (16, 8),
        "dropout": 0.05,
        "learning_rate": 1e-3,
        "emb_size": 300,
        "epochs": 50
    }

    # Load embeddings (QID -> embedding)
    embeddings = lmdb.open(args.embeddings, readonly=True, readahead=False)
    embedding_hashmap = embeddings.begin()
    
    # Ensure sequenced_decisions is loaded correctly
    with open(args.sequenced_decisions, "rb") as f:
        sequenced_decisions = pickle.load(f)
    print(len(sequenced_decisions.keys()))
    # Loading decision file
    decisions = pandas.read_csv(args.decisions).dropna()
   
    Y=decisions["target"]
    X=decisions.drop(columns="target")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.1, random_state=42, stratify=Y
    )
    train = pandas.concat([X_train, y_train], axis=1)
    val = pandas.concat([X_val, y_val], axis=1)
   
    # Print dataset sizes
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    val_features = []
    val_labels = []

    train_features = []
    train_labels = []
            

    logger.info("Building training data")
    keeping_decisions_train = train[train["target"] == 1]
    pruning_decisions_train = train[train["target"] == 0]

      # Generate analogies from keeping and pruning decisions
    get_analogies(train_features,
                        train_labels, 
                        keeping_decisions_train[["from", "QID", "depth", "starting label", "label"]], 
                        pruning_decisions_train[["from", "QID", "depth", "starting label", "label"]], 
                        params["nb_training_analogies_per_decision"],
                        sequenced_decisions,
                        params["sequence_length"],
                        params["padding"],
                        params["analogical_properties"],
                        params["valid_analogies_pattern"],
                        params["invalid_analogies_pattern"],
                        embedding_hashmap,
                        params["knn"])

    logger.info("Building validation analogies")
    keeping_decisions_val = val[val["target"] == 1]
    pruning_decisions_val = val[val["target"] == 0]

            # Generate analogies from keeping and pruning decisions
    get_testing_analogies(val_features,
                                val_labels,
                                keeping_decisions_val[["from", "QID", "depth", "starting label", "label"]].values.tolist(), 
                                pruning_decisions_val[["from", "QID", "depth", "starting label", "label"]].values.tolist(),
                                keeping_decisions_train[["from", "QID", "depth", "starting label", "label"]],
                                pruning_decisions_train[["from", "QID", "depth", "starting label", "label"]],
                                params["nb_test_analogies"],
                                sequenced_decisions,
                                params["sequence_length"],
                                params["padding"],
                                params["analogical_properties"],
                                params["valid_analogies_pattern"],
                                params["invalid_analogies_pattern"],
                                embedding_hashmap,
                                params["knn"])
                    
    val_features = np.transpose(val_features, (0, 2, 1, 3))
    val_labels = np.array(val_labels)
    
    train_features = np.transpose(train_features, (0, 2, 1, 3))
    train_labels = np.array(train_labels)
   
    def objective(trial):
        # Hyperparameter search space
        seq_len = 4
        emb_size = params["emb_size"]
        nb_filters1 = trial.suggest_categorical("nb_filters1", [8, 16, 32,64,128])
        nb_filters2 = trial.suggest_categorical("nb_filters2", [8, 16, 32,64])
        dropout = trial.suggest_float("dropout", 0, 0.6)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(skf.split(X, Y))

        test_acc_scores = []

        for fold in tqdm.tqdm(range(5), desc="Fold Iteration"):  # Iterate over each fold
            # Set the validation set to the next testing fold
            val_set_fold = set(folds[(fold + 1) % 5][1])
            val_features_fold, val_labels_fold = [], []

            # Define the training set by excluding the validation set from the current fold's training set
            train_set_fold = set(folds[fold][0]) - val_set_fold  
            train_features_fold, train_labels_fold = [], []

            # Get the test set for the current fold
            test_set_fold = set(folds[fold][1])
            test_features_fold, test_labels_fold = [], []

            X_train_fold, X_val_fold, X_test_fold = X.iloc[list(train_set_fold)], X.iloc[list(val_set_fold)], X.iloc[list(test_set_fold)]
            Y_train_fold, Y_val_fold, Y_test_fold = Y.iloc[list(train_set_fold)], Y.iloc[list(val_set_fold)], Y.iloc[list(test_set_fold)]
            
            # Concatenate features and labels
            train_fold = pandas.concat([X_train_fold, Y_train_fold], axis=1)
            val_fold = pandas.concat([X_val_fold, Y_val_fold], axis=1)
            test_fold = pandas.concat([X_test_fold, Y_test_fold], axis=1)
        
            logger.info("Building training data")
            keeping_decisions_train_fold = train_fold[train_fold["target"] == 1]
            pruning_decisions_train_fold = train_fold[train_fold["target"] == 0]
            get_analogies(train_features_fold,
                        train_labels_fold,
                        keeping_decisions_train_fold[["from", "QID", "depth", "starting label", "label"]], 
                        pruning_decisions_train_fold[["from", "QID", "depth", "starting label", "label"]],
                        params["nb_training_analogies_per_decision"],
                        sequenced_decisions,
                        params["sequence_length"],
                        params["padding"],
                        params["analogical_properties"],
                        params["valid_analogies_pattern"],
                        params["invalid_analogies_pattern"],
                        embedding_hashmap,
                        params["knn"])

            logger.info("Building validation analogies")
            keeping_decisions_val_fold = val_fold[val_fold["target"] == 1]
            pruning_decisions_val_fold = val_fold[val_fold["target"] == 0]
            get_testing_analogies(val_features_fold,
                                val_labels_fold,
                                keeping_decisions_val_fold[["from", "QID", "depth", "starting label", "label"]].values.tolist(),
                                pruning_decisions_val_fold[["from", "QID", "depth", "starting label", "label"]].values.tolist(),
                                keeping_decisions_train_fold[["from", "QID", "depth", "starting label", "label"]],
                                pruning_decisions_train_fold[["from", "QID", "depth", "starting label", "label"]],
                                params["nb_test_analogies"],
                                sequenced_decisions,
                                params["sequence_length"],
                                params["padding"],
                                params["analogical_properties"],
                                params["valid_analogies_pattern"],
                                params["invalid_analogies_pattern"],
                                embedding_hashmap,
                                params["knn"])
            
          
            logger.info("Building testing analogies")
            keeping_decisions_test_fold = test_fold[test_fold["target"] == 1]
            pruning_decisions_test_fold = test_fold[test_fold["target"] == 0]
            
            get_testing_analogies(test_features_fold,
                                test_labels_fold,
                                keeping_decisions_test_fold[["from", "QID", "depth", "starting label", "label"]].values.tolist(),
                                pruning_decisions_test_fold[["from", "QID", "depth", "starting label", "label"]].values.tolist(),
                                keeping_decisions_train_fold[["from", "QID", "depth", "starting label", "label"]],
                                pruning_decisions_train_fold[["from", "QID", "depth", "starting label", "label"]],
                                params["nb_test_analogies"],
                                sequenced_decisions,
                                params["sequence_length"],
                                params["padding"],
                                params["analogical_properties"],
                                params["valid_analogies_pattern"],
                                params["invalid_analogies_pattern"],
                                embedding_hashmap,
                                params["knn"])
            
            # Reshape the data
            val_features_fold = np.transpose(val_features_fold, (0, 2, 1, 3))
            val_labels_fold = np.array(val_labels_fold)
            train_features_fold = np.transpose(train_features_fold, (0, 2, 1, 3))
            train_labels_fold = np.array(train_labels_fold)
            test_features_fold = np.transpose(test_features_fold, (0, 2, 1, 3))
            test_labels_fold = np.array(test_labels_fold)
            print(np.shape(train_features_fold))
            print(np.shape(val_features_fold))
            print(np.shape(test_features_fold))

            # Create model
            model_fold = sequenced_analogy_model(seq_len, nb_filters1, nb_filters2, dropout, emb_size)

            # Compile model
            model_fold.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss=tf.keras.losses.BinaryCrossentropy(),
                            metrics=[tf.keras.metrics.BinaryAccuracy()])

            # Compute class weights dynamically for imbalanced data
            weight_for_0 = 1 / np.count_nonzero(train_labels_fold == 0) * (len(train_labels_fold) / 2.0)
            weight_for_1 = 1 / np.count_nonzero(train_labels_fold == 1) * (len(train_labels_fold) / 2.0)

            # Early stopping callback
            early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")

            # Train model
            history = model_fold.fit(train_features_fold, train_labels_fold,
                                    validation_data=(val_features_fold, val_labels_fold),
                                    batch_size=32, epochs=20, callbacks=[early_stopping_cb],
                                    class_weight={0: weight_for_0, 1: weight_for_1},
                                    shuffle=True, verbose=0)

            loss,accuracy = model_fold.evaluate(test_features_fold, test_labels_fold,verbose=0,batch_size=32)
            test_acc_scores.append(accuracy)
        print("TRIAL\n",flush=True)
        print(np.mean(test_acc_scores),flush=True)
        print(np.std(test_acc_scores),flush=True)
        # Return the average test accuracy across all folds
        return np.mean(test_acc_scores)


    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

        # Print best hyperparameters
    print(f"Best hyperparameters:", study.best_params)

        # Train final model using best hyperparameters
    best_params = study.best_params
    sequenced_analogy_classifier = sequenced_analogy_model(
    params["sequence_length"], best_params["nb_filters1"], 
    best_params["nb_filters2"], best_params["dropout"], params["emb_size"])
    
    
        # Compute class weights
    weight_for_0 = 1 / np.count_nonzero(train_labels== 0) * (len(train_labels) / 2.0)
    weight_for_1 = 1 / np.count_nonzero(train_labels == 1) * (len(train_labels) / 2.0)

    sequenced_analogy_classifier.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
            metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.BinaryAccuracy()]
        )

        # Early stopping callback
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, monitor="val_loss"
        )

        # Train model
    sequenced_analogy_classifier.fit(
            train_features,
            train_labels,
            validation_data=(val_features, val_labels),
            callbacks=[early_stopping_cb],
            class_weight={0: weight_for_0, 1: weight_for_1},
            shuffle=True,
            epochs=params["epochs"]
        )

    
    sequenced_analogy_classifier.save(args.model)

    
if __name__ == '__main__':
    main()
