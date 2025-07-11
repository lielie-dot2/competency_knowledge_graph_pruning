# Pair-based Competency for Analogical Reasoning: Knowledge Graph Pruning

This repository contains code and datasets to reproduce the experiments presented in the paper:

**"Which Pairs to Choose? Exploring Analogical Competency for Knowledge Graph Pruning"**

---

## Overview
This works present the task of knowledge graph entity pruning through an analogical detection problem as done in the paper ([zero shot boostrapping](https://arxiv.org/pdf/2306.16296)). It explores various ****selection methodologies to identify the most competent pairs**** to use in analogical inference, based on the embedding representations of entity pairs. and a labeled set of entity pairs.

---
## Datasets

Experiments are performed on three labeled datasets containing pairs labeled with **prune (0)** and **keep (1)** decisions derived from [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page), each representing a distinct domain:

| **Dataset**        | **Domain**           | **Description**                                        | **Resource**                                        |
| ------------------ | -------------------- | ------------------------------------------------------ | --------------------------------------------------- |
| **DS1** (dataset1) | Computer Science     | Entity pairs related to computer science concepts      | [Zenodo - DS1](https://zenodo.org/records/8091584)  |
| **DS2** (dataset2) | General Knowledge    | Entity pairs across various general knowledge topics   | [Zenodo - DS2](https://zenodo.org/records/8091584)  |
| **DS3** (dataset3) | Art and Works of Art | Entity pairs related to artworks | [Zenodo - DS3](https://zenodo.org/records/15600971) |

## Methodologies for Pair Selection

Following the work of [Bounhas et al.](https://arxiv.org/pdf/2306.16296), competency is assessed from the perspective of the **vector embeddings** of entity pairs. Since pairs are represented by concatenations or combinations of individual entity embeddings, several distance-based methodologies are used to estimate how "close" a training pair is to a test pair:

### Distance-Based Methods:

The following selection methodologies are used:

- **Random:** Random retrieval
- **M1:** Average Euclidean distance between corresponding elements.
- **M2:** Distance between heads only.
- **M3:** Distance between tails only.
- **M4:** Distance between barycenters (midpoints) of pairs.
- **M5:** Based on minimizing the **log-loss** of analogical inference predictions, selecting pairs that lead tothe lowest prediction error. 

## Repository Structure
- `./data/` — Datasets and train/test splits
- `./models/` — Analogical classifiers trained on the different models
- `./src/` — Scripts to run experiments
- `./src/selection_methods.py` — Implementations of the different selection methodologies
- `./src/find_best_pairs.py` — Main script to run all experimentations using the models, datasets and methods
- `./src/generate_sequence_simple.py` —  Script that generates a dictionnary with the concatenation of the embeddings of pairs of entities in the decision file
- `./src/model_tuning_training.py` — Script to tune and train the analogical model on a particular Dataset
- `./src/distance_calculation.py` — Utilities for selection methodologies
- `./src/embeddings_utils.py` — Utilities for embedding processing

### Environment
- [Python 3.x](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
- Install required Python packages using pip and the file `requirements.txt` 

## Requirements
 Wikidata Embedding hashmap as an [lmdb](https://lmdb.readthedocs.io/en/release/) database. You can use either [PyTorch-BigGraph](https://torchbiggraph.readthedocs.io/en/latest/)or [Pykeen](https://pykeen.readthedocs.io/en/stable/api/pykeen.pipeline.pipeline.html#pykeen.pipeline.pipeline). You can freely adjust the training setup by modifying the configuration file located at: `./configuration/config.yaml` 

## Step 1: Clone the repository   
First, you will have to clone the repository on Github and go to the section called 
```bash
git clone https://github.com/lielie-dot2/competency_knowledge_graph_pruning.git
```
## Step 2: Generate the sequenced decisions dictionnary
```bash
python3 -m src/generate_sequence_simple.py --decisions1 ./data/dataset1_gold_decisions.csv --decisions2 ./data/dataset2_gold_decisions.csv --decisions3 ./data/dataset3_gold_decisions.csv --output ./data/sequenced_simple.pkl --embeddings path/to/your/embedding/hashmap/database
```
## Step 3: Tune and Train the analogical model on a datasets
```bash
python3 -m src/model_tuning_training.py --decisions ./data/dataset1_gold_decisions.csv --model ./models/model_dataset1.h5 --embeddings path/to/your/embedding/hashmap/database --sequenced-decisions ./data/sequenced_simple.pkl
```
```bash
python3 -m src/model_tuning_training.py --decisions ./data/dataset2_gold_decisions.csv --model ./models/model_dataset2.h5 --embeddings path/to/your/embedding/hashmap/database --sequenced-decisions ./data/sequenced_simple.pkl
```
```bash
python3 -m src/model_tuning_training.py --decisions ./data/dataset3_gold_decisions.csv --model ./models/model_dataset3.h5 --embeddings path/to/your/embedding/hashmap/database --sequenced-decisions ./data/sequenced_simple.pkl
```
## Step 4: Pre-compute (M5) scores
```bash
python3 -m src/relative_competence.py --config ./configuration/config.yml
```
## Step 5: Launch the main script for all datasets and methodologies
```bash
python3 -m src/find_best_pairs.py --config ./configuration/config.yml
```
## Step 6: Launch the main script for all datasets and methodologies

Inspect and Analyze the results `/data/optimal_results_all_methods.csv"`

## Credits

Some materials used in this project are adapted from resources provided by [analogical-pruning](https://github.com/Orange-OpenSource/analogical-pruning) Github repository. The files that have been used and modified are clearly marked with a license header, in accordance with the original repository license.
We acknowledge the original authors for making their work publicly available.