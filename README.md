# Dual Attention for Limit Order Book Price Trend Prediction

This repository contains the research code for a thesis project on price trend
prediction from Limit Order Book (LOB) data. I use it as an experimental
framework to study how event-level preprocessing, continuous-time feature
representations, and transformer-based architectures can be combined for
short-horizon market movement prediction.

The project is not intended to be a polished trading system. It is a research
prototype designed to make modelling assumptions explicit, test alternative LOB
representations, and evaluate whether richer event-window features improve
classification of future price trends.

## Research Motivation

LOB data is high-frequency, irregular, noisy, and strongly event-driven. A core
question in this project is whether I can represent local LOB dynamics in a way
that is more informative than raw snapshots alone, while still preserving the
temporal structure needed by a transformer model.

I focus on two complementary research directions:

1. **Preprocessing and token construction for LOB events**

   I explore dynamic and static transformations of LOB windows before they are
   passed to the model. For dynamic streams, I experiment with a spline-based
   kinematic representation inspired by **["Kinematic Tokenization:
   Optimization-Based Continuous-Time Tokens for Learnable Decision Policies in
   Noisy Time Series"](https://arxiv.org/abs/2601.09949)**. In practice, this means extracting continuous-time
   position, velocity, acceleration, and jerk-style features from price and
   volume trajectories.

   For static LOB state features, I use PLGS-style price scaling and exponential
   volume scaling inspired by **["LOBERT: Generative AI Foundation Model for Limit
   Order Book Messages"](https://arxiv.org/abs/2511.12563)**. This gives the preprocessing pipeline a way to encode
   the relative structure of the book while reducing the effect of raw scale.

2. **Dual-attention modelling for price trend prediction**

   I implement a transformer-style architecture inspired by **"TLOB: A Novel
   Transformer Model with Dual Attention for Price Trend Prediction with Limit
   Order Book Data"**. The model combines feature-wise attention over LOB-derived
   variables with temporal attention over event sequences. The goal is to study
   whether separating feature interactions and temporal dependencies is useful
   for LOB trend classification.

## Current Pipeline

The codebase is organized around a two-stage workflow.

```bash
python scripts\process_data.py
python scripts\run_training.py
```

`scripts/process_data.py` runs the preprocessing pipeline. It loads
LOBSTER-style message and orderbook files, joins them, filters the trading
session, creates trend labels, computes static and kinematic features,
normalizes derivative features on the training split, and saves both processed
dataframes and model-ready sequence tensors.

`scripts/run_training.py` loads the generated sequence tensors, creates PyTorch
datasets and dataloaders, instantiates the dual-attention model, and trains it
using the configured training loop.

Each training sample is a full event window of length `sequence_window`, defined
in `configs/pipeline_config.yaml`.

## Repository Structure

```text
.
|-- configs/
|   `-- pipeline_config.yaml              # Main experiment, preprocessing, model, and training config
|-- data/
|   |-- LOBSTER/                          # Raw LOBSTER-style files, ignored by git
|   |-- processed_dataframes/             # Processed CSV outputs
|   |-- sequences/                        # Saved X/T/y NumPy sequence tensors
|   `-- derivatives_z_scores/             # Fitted derivative normalization statistics
|-- logs/                                 # Contains logging files
|-- results/                              # Trained model checkpoints and experimental outputs
|-- scripts/
|   |-- vram_dry_run.py                   # GPU VRAM workload test on HPC
|   |-- process_data.py                   # Runs the preprocessing pipeline
|   `-- run_training.py                   # Trains the model from saved sequences
|-- src/
|   |-- configuration.py                  # YAML configuration dataclasses
|   |-- datasets.py                       # Sequence construction and PyTorch dataset
|   |-- fast_kinematic_preprocessing.py   # Faster vectorized alternative to compute kinematic stream
|   |-- horizon.py                        # Trend labelling strategies
|   |-- kinematic_preprocessing.py        # Static and kinematic LOB feature engineering
|   |-- model.py                          # Dual-attention transformer model
|   |-- processing.py                     # End-to-end preprocessing pipeline
|   |-- run_logging.py                    # Logging functions
|   |-- training.py                       # Loss, trainer, optimizer loop
|   `-- utils.py                          # YAML and helper utilities
|-- tests/
|   |-- unit/                             # Focused tests for small components
|   |-- integration/                      # Synthetic pipeline/model integration tests
|   `-- smoke/                            # Smoke test on a small LOBSTER sample
|-- requirements.txt                      
|-- dry-run.pbs                           # Dry-run test on HPC 
|-- env_setup.sh                          # Allows to setup the required conda environment
|-- environment.yml                       # Contains package and versions to create conda env
|--run_script.pbs                         # PBS script to run full experiment on HPC cluster
`-- pytest.ini
```

## Data Products

The preprocessing stage writes normalized processed dataframes to:

```text
data/processed_dataframes/<split>/<symbol>_<date>_processed.csv
```

It writes model-ready sequence tensors to:

```text
data/sequences/<split>/<symbol>_<date>_features.npy
data/sequences/<split>/<symbol>_<date>_times.npy
data/sequences/<split>/<symbol>_<date>_labels.npy
```

`LOBDataset` reconstructs sliding windows from these compact arrays during training.

## Testing

I maintain unit and integration tests to keep the research pipeline stable while
I iterate on modelling ideas.

```bash
python -m pytest -q
```

On my local machine, I usually run the tests through the `transformer` conda
environment:

```bash
conda --no-plugins run -n transformer python -m pytest -q
```

The smoke test in `tests/smoke/` is used as a lightweight end-to-end sanity
check on a small LOBSTER sample.

## HPC Runs guidelines

In order to launch a run from the HPC, you may execute the following: 

```bash
cd $HOME

git clone https://github.com/iliasra/Dual-Attention-for-LOB-Price-Trend-Prediction.git

chmod +x Dual-Attention-for-LOB-Price-Trend-Prediction/env_setup.sh

./Dual-Attention-for-LOB-Price-Trend-Prediction/env_setup.sh Dual-Attention-for-LOB-Price-Trend-Prediction/environment.yml

qsub Dual-Attention-for-LOB-Price-Trend-Prediction/run_script.pbs
```

## Status

This repository is under active development as part of my thesis. The current
implementation provides the core preprocessing, sequence construction,
dual-attention model, and training loop. The main ongoing work is experimental:
I am refining the feature representations, validating the labelling choices, and
comparing model behaviour across alternative LOB tokenization strategies.

Once a first run is performed on this particular architecture, I would like to implement and test the following ideas: work on a volume-based view of the data (i.e. one snapshot each x traded shares); switch temporal/spatial attentions execution order, to see what impact the order has; implement an adaptative horizon framework based on the paper ["The Label Horizon Paradox: Rethinking Supervision Targets in Financial Forecasting"](https://arxiv.org/abs/2602.03395); try to apply kinematic processing to a more continuous feature, such as microprice, as raw prices/volumes may be too discrete for the method to perform well.   
