# Pair-based Competency for Analogical Reasoning: Knowledge Graph Pruning

This repository contains code and datasets to reproduce the experiments presented in the paper:

**"Which Pairs to Choose? Exploring Analogical Competency for Knowledge Graph Pruning"**

---

## Overview
This works present the task of knowledge graph entity pruning through an analogical detection problem as done in the paper ([zero shot boostrapping](https://arxiv.org/pdf/2306.16296)). It explores various ****selection methodologies to identify the most competent pairs**** to use in analogical inference, based on the embedding representations of entity pairs. and a labeled set of entity pairs.

---
## Datasets

Experiments are performed on three labeled datasets derived from [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page), each representing a distinct domain:

| Dataset | Domain | Description |
| --- | --- | --- |
| DS1 (dataset1) | Computer Science | Entity pairs related to computer science concepts |
| DS2 (dataset2) | General Knowledge | Broad domain entity pairs across various topics |
| DS3 (dataset3) | Art and Works of Art | Entity pairs related to artistic works and artists |


Each dataset contains pairs labeled with **prune (0)** or **keep (1)** decisions indicating whether the pair should be kept for analogy inference. The data and train/test splits are located in the `/data` directory.
---

## Methodologies for Pair Selection

Following [13], competency is assessed from the perspective of the **vector embeddings** of entity pairs. Since pairs are represented by concatenations or combinations of individual entity embeddings, several distance-based methodologies are used to estimate how "close" a training pair is to a test pair:

### Distance-Based Methods:

Given two pairs (a,b)(a, b)(a,b) and (c,d)(c, d)(c,d), with embeddings a,b,c,da, b, c, da,b,c,d, the following distance metrics are used:

- **M1:** Average Euclidean distance between corresponding elements.
- **M2:** Distance between heads only.
- **M3:** Distance between tails only.
- **M4:** Distance between barycenters (midpoints) of pairs.
- **M5:** Based on minimizing the **log-loss** of analogical inference predictions, selecting pairs that lead tothe lowest prediction error. 

## Repository Structure
- `/data/` — Datasets and train/test splits
- `/models/` — Analogical classifiers trained on the different models
- `/src/` — Scripts to run experiments
- `/selection_methods.py` — Implementations of the different selection methodologies
- `/find_best_pairs.py` — Main script to run all experimentations using the models, datasets and methods
- `/distance_calculation.py` — Utilities for selection methodologies
- `/embeddings_utils.py` — Utilities for embedding processing

### Environement
- [Python 3.x](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
- Install required Python packages using pip and the file `requirements.txt` 

## Step 1: Clone the repository   
First, you will have to clone the repository on Github and go to the section called 
```bash
git clone https://github.com/lielie-dot2/competency_knowledge_graph_pruning.git
```
## Step 2: Pre-compute (M5) scores
```bash
python3 -m src/relative_competence.py --config ./configuration/config.yml
```
## Step 3: Launch the main script for all datasets and methodologies
```bash
python3 -m src/find_best_pairs.py --config ./configuration/config.yml
```
## Step 4: Launch the main script for all datasets and methodologies

Inspect and Analyze the results `/data/optimal_results_all_methods.csv"`
