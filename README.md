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

The codebase is organized around a preprocessing and training workflow. When fast kinematic tokenization is enabled, an optional GCV cache can be built before preprocessing so that expanding-window folds do not recompute the same daily GCV scores repeatedly.

```bash
python scripts/process_data.py
python scripts/run_training.py
```

`scripts/process_data.py` runs the preprocessing pipeline. It loads
LOBSTER-style message and orderbook files, joins them, filters the trading
session, creates trend labels, computes static and kinematic features,
normalizes derivative features on the training split, and saves model-ready
sequence tensors. Large processed dataframe CSVs are optional and controlled by
`preprocessing.save_processed_dataframes`.

`scripts/run_training.py` loads the generated sequence tensors, creates PyTorch
datasets and dataloaders, instantiates the dual-attention model, and trains it
using the configured training loop.

For fast kinematic tokenization, `scripts/list_lambda_gcv_tasks.py` and
`scripts/build_lambda_gcv_cache.py` can precompute daily GCV score caches under `data/gcv_cache/`. The preprocessing pipeline can then read those caches through `--lambda-cache-dir`.

Each training sample is a full event window of length `sequence_window`, defined
in `configs/pipeline_config.yaml`.

## Repository Structure

```text
.
|-- configs/
|   |-- folds.txt                         # Fold ids to preprocess in PBS array jobs
|   |-- lobster_column_schema.yaml        # Explicit LOBSTER message/orderbook column schema
|   `-- pipeline_config.yaml              # Main experiment, preprocessing, model, and training config
|-- data/
|   |-- LOBSTER/                          # Raw LOBSTER-style files, ignored by git
|   |-- gcv_cache/                        # Daily GCV lambda cache files and task lists
|   |-- processed_dataframes/             # Optional processed CSV outputs
|   |-- sequences/                        # Saved X/T/y NumPy sequence tensors
|   `-- derivatives_z_scores/             # Fitted derivative normalization statistics
|-- logs/                                 # Run logs, metrics, config snapshots, PBS logs
|-- results/                              # Trained model checkpoints
|-- scripts/
|   |-- build_lambda_gcv_cache.py         # Builds one daily GCV cache task
|   |-- list_lambda_gcv_tasks.py          # Lists daily GCV cache tasks
|   |-- process_data.py                   # Runs the preprocessing pipeline
|   |-- run_training.py                   # Trains the model from saved sequences
|   |-- validate_lobster_format.py        # Validates raw LOBSTER CSV files
|   `-- vram_dry_run.py                   # GPU VRAM workload test on HPC
|-- src/
|   |-- configuration.py                  # YAML configuration dataclasses
|   |-- datasets.py                       # Sequence construction and PyTorch dataset
|   |-- fast_kinematic_preprocessing.py   # Faster vectorized alternative to compute kinematic stream
|   |-- gcv_lambda_cache.py               # Daily GCV cache construction and aggregation
|   |-- horizon.py                        # Trend labelling strategies
|   |-- kinematic_preprocessing.py        # Static and kinematic LOB feature engineering
|   |-- lobster_io.py                     # LOBSTER CSV loading and column inference
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
|-- build_lambda_gcv_cache_array.pbs       # PBS array job for GCV cache construction
|-- preprocess_folds_array.pbs             # PBS array job for fold-level preprocessing
|-- run_training.pbs                       # PBS script to train from preprocessed folds
`-- pytest.ini
```

## Data Products

When preprocessing folds are enabled, outputs are fold-scoped.

If `preprocessing.save_processed_dataframes: true`, the preprocessing stage also
writes normalized processed dataframes to:

```text
data/processed_dataframes/<fold_id>/<split>/<symbol>_<date>_processed.csv
```

By default this option is `false` to save disk space. The training stage does
not need these CSVs. It reads the model-ready sequence tensors written to:

```text
data/sequences/<fold_id>/<split>/<symbol>_<date>_features.npy
data/sequences/<fold_id>/<split>/<symbol>_<date>_times.npy
data/sequences/<fold_id>/<split>/<symbol>_<date>_labels.npy
```

`LOBDataset` reconstructs sliding windows from these compact arrays during training.

Additional preprocessing artifacts include:

```text
data/sequences/<fold_id>/preprocessing_metadata.yaml
data/sequences/<fold_id>/feature_schema.yaml
data/derivatives_z_scores/<fold_id>/derivatives_stats.yaml
```

If fast kinematic tokenization uses the daily GCV cache, cache artifacts are
stored as:

```text
data/gcv_cache/lambda_gcv_tasks.txt
data/gcv_cache/<cache_key>/<price|volume>/<symbol>_<date>.npz
```

Training writes per-run and per-fold logs/checkpoints under the directories
configured by `data.logs_dir` and `training.model_dir` (default: `logs/` and
`results/`). The run directory is named from `TRAINING_RUN_STEM` when provided,
or from `experiment.name` plus the launch timestamp in the PBS scripts.

```text
logs/<RUN_STEM>/summary_<fold_id>.yaml
logs/<RUN_STEM>/<fold_id>/run.log
logs/<RUN_STEM>/<fold_id>/metrics.csv
logs/<RUN_STEM>/<fold_id>/confusion_matrices.yaml
logs/<RUN_STEM>/<fold_id>/probabilities/
results/<RUN_STEM>/<fold_id>/best_lob_transformer.pth
results/<RUN_STEM>/<fold_id>/training_state_latest.pth
```

Optional Weights & Biases tracking is configured under `tracking.wandb` and is
disabled by default. To enable it on a machine with network access, set
`tracking.wandb.enabled: true` in the config and provide credentials outside the
repo:

```bash
export WANDB_API_KEY=<your_key>
export WANDB_PROJECT=lob-price-trend
export WANDB_DIR="$PROJECT_DIR/logs/wandb"
```

For compute nodes without outbound network access, run with offline mode and
sync after the job:

```bash
export WANDB_MODE=offline
wandb sync --sync-all <wandb_run_or_directory>
```

Training can resume from the complete state checkpoint with:

```bash
python scripts/run_training.py --config configs/pipeline_config.yaml --resume-latest
python scripts/run_training.py --config configs/pipeline_config.yaml --fold-id <fold_id> --resume-from <path/to/training_state_latest.pth>
```

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

## HPC Runs Guidelines

The expected HPC workflow is:

1. create the environment;
2. generate and build the daily GCV cache;
3. preprocess folds as array jobs;
4. train the model from the preprocessed fold artifacts.

Start by cloning the project and creating the conda environment:

```bash
cd $HOME

git clone https://github.com/iliasra/Dual-Attention-for-LOB-Price-Trend-Prediction.git

chmod +x Dual-Attention-for-LOB-Price-Trend-Prediction/env_setup.sh

./Dual-Attention-for-LOB-Price-Trend-Prediction/env_setup.sh Dual-Attention-for-LOB-Price-Trend-Prediction/environment.yml
```

Place the raw LOBSTER files under:

```text
$HOME/Dual-Attention-for-LOB-Price-Trend-Prediction/data/LOBSTER/
```

Then edit `configs/pipeline_config.yaml` and `configs/folds.txt` as needed. If
you use `preprocess_folds_array.pbs`, its `#PBS -J` array range must match the
number of non-empty, non-comment lines in `configs/folds.txt`, or you must
override it at submission time with `qsub -J ...`. The same rule applies to
`run_training_folds_array.pbs` when training folds as an array job.

`configs/folds.txt` is only the list of fold ids to run. Each non-empty,
non-comment line contains one fold id, and every listed id must exist in the
`folds:` section of the YAML config. The actual train/validation/test dates are
defined in the YAML, not in `folds.txt`.

Example:

```text
fold_001
fold_002
# fold_003 is skipped
fold_004
```

For example:

```bash
N_FOLDS=$(awk 'NF && $1 !~ /^#/ { c++ } END { print c + 0 }' configs/folds.txt)
qsub -J 1-$N_FOLDS preprocess_folds_array.pbs
```

If there is only one fold and your PBS rejects `1-1`, use:

```bash
qsub -J 1 preprocess_folds_array.pbs
```

### 1. Generate GCV Cache Tasks

From the project root:

```bash
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate transformer

cd $HOME/Dual-Attention-for-LOB-Price-Trend-Prediction

PYTHONPATH="$PWD/src:${PYTHONPATH:-}" \
python scripts/list_lambda_gcv_tasks.py
```

This prints the number of cache tasks and writes:

```text
data/gcv_cache/lambda_gcv_tasks.txt
```

If `preprocessing.kinematic_tokenization.method` is not `fast`, the task file is
empty and this cache step can be skipped.

### 2. Build the Daily GCV Cache

Submit one PBS array task per line of `data/gcv_cache/lambda_gcv_tasks.txt`.
Replace `<N_TASKS>` with the number printed by `list_lambda_gcv_tasks.py`.

```bash
qsub -J 1-<N_TASKS> build_lambda_gcv_cache_array.pbs
```

Each task stages the raw files for one `(symbol, date, price|volume)` job and
writes `.npz` cache files under `data/gcv_cache/`.

### 3. Preprocess Folds

Once the GCV cache is complete, submit the fold preprocessing array:

```bash
qsub preprocess_folds_array.pbs
```

If the `#PBS -J` directive in the script does not match your fold count, submit
with an explicit array range instead:

```bash
qsub -J 1-<N_FOLDS> preprocess_folds_array.pbs
```

`preprocess_folds_array.pbs` reads fold ids from `configs/folds.txt`, runs:

```bash
python scripts/process_data.py \
  --fold-id "$FOLD_ID" \
  --lambda-cache-dir "$PROJECT_DIR/data/gcv_cache" \
  --require-lambda-cache
```

and copies fold outputs back to:

```text
data/sequences/<fold_id>/
data/processed_dataframes/<fold_id>/   # only when save_processed_dataframes=true
data/derivatives_z_scores/<fold_id>/
```

For a local or sequential preprocessing run without the cache requirement, use:

```bash
python scripts/process_data.py
```

Without `--fold-id` or `--fold-index`, all configured folds are processed
sequentially.

### 4. Train

After preprocessing has produced all configured fold sequence directories:

To train folds independently as PBS array jobs, use `run_training_folds_array.pbs`.
Each array task reads one fold id from `configs/folds.txt`. As with
preprocessing, the array range must match the number of non-empty, non-comment
lines in the folds file, or be overridden at submission time.

```bash
N_FOLDS=$(awk 'NF && $1 !~ /^#/ { c++ } END { print c + 0 }' configs/folds.txt)
RUN_STEM=my_experiment_$(date +%Y%m%d_%H%M%S)
qsub -J 1-$N_FOLDS \
  -v TRAINING_RUN_STEM="$RUN_STEM" \
  run_training_folds_array.pbs
```

For a non-default config or folds file, pass them explicitly:

```bash
qsub -J 1-<N_FOLDS> \
  -v TRAINING_CONFIG=configs/my_config.yaml,FOLDS_FILE=configs/my_folds.txt,TRAINING_RUN_STEM="$RUN_STEM" \
  run_training_folds_array.pbs
```

For a sequential training run that processes the fold ids one after another in a
single PBS job, use:

```bash
qsub run_training.pbs
```

`run_training.pbs` trains over the selected folds sequentially and copies
generated logs and model weights back into `$PROJECT_DIR/logs/` and
`$PROJECT_DIR/results/`. It also saves the PBS stdout/stderr stream under:

```text
logs/<RUN_STEM>/pbs/run_logs.txt
```

and mirrors that stream under:

```text
$HOME/run_outputs/<RUN_STEM>/run_logs.txt
```

For `run_training_folds_array.pbs`, each fold task writes its live PBS stream to:

```text
logs/<RUN_STEM>/<fold_id>/run_logs.txt
```

## Status

This repository is under active development as part of my thesis. The current
implementation provides the core preprocessing, sequence construction,
dual-attention model, and training loop. The main ongoing work is experimental:
I am refining the feature representations, validating the labelling choices, and
comparing model behaviour across alternative LOB tokenization strategies.

Once a first run is performed on this particular architecture, I would like to implement and test the following ideas: work on a volume-based view of the data (i.e. one snapshot each x traded shares); switch temporal/spatial attentions execution order, to see what impact the order has; implement an adaptative horizon framework based on the paper ["The Label Horizon Paradox: Rethinking Supervision Targets in Financial Forecasting"](https://arxiv.org/abs/2602.03395); try to apply kinematic processing to a more continuous feature, such as microprice, as raw prices/volumes may be too discrete for the method to perform well.   
