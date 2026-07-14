# Economically Grounded Evaluation for LOB Price Trend Prediction

This repository contains the research code for a thesis project on price trend
prediction from Limit Order Book (LOB) data. The objective is to build and test an
evaluation framework for price trend prediction where labels, thresholds,
metrics, and model selection are tied to economically meaningful trading
signals.

A dual-attention transformer architecture is used as a case study inside this framework. It
provides a controlled model family for testing hypotheses about event-level LOB
representations, continuous-time kinematic streams, local attention over noisy
sequences, and optional mixture-of-experts (MoE) routing. The broader research
question is whether a model that looks good under standard classification
metrics still produces useful directional signals once the labelling and
evaluation protocol accounts for spread, execution costs, calibration, and
fixed-rate signal selection.

The project is not intended to be a polished trading system. It is a research
prototype designed to make modelling assumptions explicit, compare economically
motivated labelling choices, and evaluate short-horizon price trend prediction in
a way that is closer to the decisions a downstream trading or backtesting system
would actually consume.

## Research Motivation

LOB data is high-frequency, irregular, noisy, and strongly event-driven. A core
problem is that conventional price-movement labels can be statistically
convenient while remaining economically weak: they may ignore spread, execution
costs, output rate, and the difference between ranking the best signals and
classifying every event. This repository therefore focuses on the full
evaluation protocol, not only the neural architecture.

I focus on three complementary research directions:

1. **Economically meaningful labels and evaluation**

   The preprocessing pipeline supports adaptive price trend labels whose
   threshold can combine estimated round-trip costs and local volatility. Model
   selection can monitor metrics such as directional precision at a fixed signal
   rate, where the top `x%` up and down probability scores are evaluated as the
   actionable signals. Post-training evaluation can fit directional thresholds
   on validation probabilities and apply the selected thresholds to held-out
   test data.

2. **Preprocessing and token construction for noisy LOB events**

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

3. **Dual-attention modelling as a case study**

   I implement a transformer-style architecture inspired by **"TLOB: A Novel
   Transformer Model with Dual Attention for Price Trend Prediction with Limit
   Order Book Data"**. The model combines feature-wise attention over LOB-derived
   variables with temporal attention over event sequences. The goal is to study
   whether separating feature interactions and temporal dependencies, optionally
   with MoE routing, is useful under the economic evaluation protocol above.

## Current Pipeline

The codebase is organized around a preprocessing, training, and evaluation
workflow. When fast kinematic tokenization is enabled, an optional GCV cache can
be built before preprocessing so that expanding-window folds do not recompute the
same daily GCV scores repeatedly.

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
datasets and dataloaders, instantiates the case-study model, and trains it using
the configured training loop.

For fast kinematic tokenization, `scripts/list_lambda_gcv_tasks.py` and
`scripts/build_lambda_gcv_cache.py` can precompute daily GCV score caches under `data/gcv_cache/`. The preprocessing pipeline can then read those caches through `--lambda-cache-dir`.

The current training process supports two supervision modes:

- `last_window`: the legacy mode, where each `sequence_window` event window
  produces one prediction for the final token.
- `token_chunk`: the current large-scale mode, where a longer chunk such as
  `sequence_window: 256` produces token-wise logits and computes the loss only
  on the supervised tail, for example the final 128 tokens. With
  `chunk_stride: 128`, each supervised event is covered once while avoiding most
  of the repeated computation from overlapping sliding windows.

In the default token-chunk configuration, attention is causal, optionally
bounded by `model.local_attention_context_tokens`, and still respects the
configured continuous-time `max_dt` mask.

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
|   |-- benchmark_token_chunk.py          # Synthetic token-chunk throughput/VRAM benchmark
|   `-- vram_dry_run.py                   # Legacy GPU VRAM workload test
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
|-- dry-run.pbs                           # Lightweight token-chunk dry-run test on HPC
|-- env_setup.sh                          # Allows to setup the required conda environment
|-- environment.yml                       # Contains package and versions to create conda env
|-- build_lambda_gcv_cache_array.pbs       # PBS array job for GCV cache construction
|-- preprocess_folds_array.pbs             # PBS array job for fold-level preprocessing
|-- run_training.pbs                       # PBS script to train from preprocessed folds
|-- run_training_folds_array.pbs           # PBS array job for fold-level training
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

The training datasets reconstruct windows or chunks from these compact arrays:

- `LOBDataset` is used in legacy `last_window` mode.
- `LOBTokenChunkDataset` is used in `token_chunk` mode. It creates per-day
  chunks, never crosses day boundaries, masks the warmup tokens, and covers each
  supervised event once according to `training.sequence_supervision.chunk_stride`.

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

Optional Weights & Biases tracking is configured under `tracking.wandb` in the
YAML. To enable it on a machine with network access, set
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

The complete training-state checkpoint stores the model, optimizer, scheduler,
AMP scaler, early-stopping state, RNG states, history, and top-k checkpoint
metadata. Classification additionally stores its validation index and W&B run
id. Action-value regression resumes at the next fully validated epoch boundary;
it intentionally does not attempt a partial-epoch replay. A checkpoint from
`last_window` training is not resumed into `token_chunk` training, and an
action-value checkpoint is not interchangeable with a classification state.

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
4. run a lightweight token-chunk GPU dry-run;
5. train the model from the preprocessed fold artifacts;
6. evaluate the selected checkpoints and thresholded signals.

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

Then edit `configs/pipeline_config.yaml` and `configs/folds.txt` as needed. PBS
array bounds are not hard-coded in the scripts: submit
`preprocess_folds_array.pbs` and `run_training_folds_array.pbs` with an explicit
`qsub -J ...` range matching the number of non-empty, non-comment lines in the
selected folds file.

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
N_FOLDS=$(awk 'NF && $1 !~ /^#/ { c++ } END { print c + 0 }' configs/folds.txt)
qsub -J 1-$N_FOLDS preprocess_folds_array.pbs
```

`preprocess_folds_array.pbs` reads fold ids from `configs/folds.txt`, runs:

```bash
python scripts/process_data.py \
  --fold-id "$FOLD_ID" \
  --lambda-cache-dir "$PROJECT_DIR/data/gcv_cache" \
  --require-lambda-cache
```

and copies fold outputs back to the directories configured by
`data.sequence_data_dir`, `data.processed_data_dir`, and
`preprocessing.normalization.derivatives_stats_dir`, for example:

```text
data/sequences/<fold_id>/
data/processed_dataframes/<fold_id>/   # only when save_processed_dataframes=true
data/derivatives_z_scores/<fold_id>/
```

Paths in the YAML are resolved relative to the YAML file itself. With the
default `configs/pipeline_config.yaml`, `../data/sequences` therefore means
`<project>/data/sequences`, not `$HOME/data/sequences`.

The train-only fitting passes use **every configured training day and every
processed row**. Days are loaded and released sequentially, so RAM usage no
longer grows with an expanding fold. Means and variances are accumulated online;
robust and tail quantiles use a deterministic bounded-memory weighted
asinh-histogram sketch updated by every finite value. The approximation method,
number of train days, and total processed row count are recorded in the
derivative-statistics metadata. Static PLGS/volume parameters and GCV
aggregation follow the same
all-train streaming principle.

This trades additional sequential passes over train for bounded memory. The PBS
preprocessing walltime is therefore set to 24 hours; peak memory is controlled
by one full day plus fixed-size per-column sketches rather than by fold length.

PBS preprocessing first removes inherited fold artifacts inside `$TMPDIR`, and
copies a fold back with `rsync --delete` only after a fresh
`sequence_manifest.yaml` exists. A successful destination contains `_SUCCESS`.
Consequently, an old destination is replaced rather than merged with stale
shards. Exit status `137` means the process was killed (normally an OOM at the
PBS memory limit); in that case nothing is copied back and `_SUCCESS` is not
created. The end of a successful log contains all of:

```text
Saved exact shard manifest for fold <fold_id>
===== COPYING OUTPUTS BACK ... =====
===== DONE <fold_id> =====
Exit status=0
```

For a local or sequential preprocessing run without the cache requirement, use:

```bash
python scripts/process_data.py
```

Without `--fold-id` or `--fold-index`, all configured folds are processed
sequentially.

### 4. Token-Chunk Dry-Run

Before submitting a long training job, run the synthetic GPU dry-run:

```bash
qsub dry-run.pbs
```

`dry-run.pbs` does not copy the raw data or sequence shards. It copies only the
code needed for the benchmark, excluding large directories such as `data/`,
`logs/`, `output/`, `results/`, and `.git/`. The job runs
`scripts/benchmark_token_chunk.py` against the current config and prints peak
CUDA memory, tokens/s, and supervised labels/s.

The benchmark uses `training.batch_size` from the YAML unless an override is
provided:

```bash
qsub -v DRY_RUN_BATCH_SIZE=64 dry-run.pbs
```

Useful dry-run overrides are:

```bash
qsub -v DRY_RUN_BATCH_SIZE=64,DRY_RUN_D_INPUT=214,DRY_RUN_STEPS=5 dry-run.pbs
```

`DRY_RUN_D_INPUT` should match the number of generated feature columns when
`model.d_input` is left as `null` in the config.

### 5. Train

After preprocessing has produced all configured fold sequence directories:

For the current workflow, `run_training.pbs` processes the fold ids from
`configs/folds.txt` sequentially in one PBS job:

```bash
RUN_STEM=my_experiment_$(date +%Y%m%d_%H%M%S)
qsub -v TRAINING_RUN_STEM="$RUN_STEM" run_training.pbs
```

For a non-default config or folds file:

```bash
qsub -v TRAINING_CONFIG=configs/my_config.yaml,FOLDS_FILE=configs/my_folds.txt,TRAINING_RUN_STEM="$RUN_STEM" \
  run_training.pbs
```

`run_training.pbs` copies the code to `$TMPDIR`, links persistent `logs/` and
`results/`, stages each fold from the configured `data.sequence_data_dir`, and
links the configured `data.raw_data_dir` into the work tree. The raw-data link is
needed by the held-out classification PnL evaluation; without it training can
finish but that diagnostic is marked as skipped. The job then runs:

`run_training.pbs` does **not** regenerate sequence shards. Whenever the label
strategy, objective, target columns, feature schema, sequence window, or sample
clock changes, rerun preprocessing before training. In particular,
`action_value_regression` requires every `*_labels.npy` file to have floating
shape `[num_rows, 2]`; scalar `[num_rows]` integer files are classification
shards and cannot be reused. Training validates the manifest and target arrays
before the first epoch and reports a stale-preprocessing error when they do not
match the active config.

For the sequential one-fold workflow, the two jobs can be chained so training
starts only after successful preprocessing:

```bash
PREPROCESS_JOB=$(qsub -v TRAINING_CONFIG=configs/pipeline_config.yaml,FOLDS_FILE=configs/folds.txt preprocess.pbs)
qsub -W depend=afterok:$PREPROCESS_JOB \
  -v TRAINING_CONFIG=configs/pipeline_config.yaml,FOLDS_FILE=configs/folds.txt,TRAINING_RUN_STEM=A5_$(date +%Y%m%d_%H%M%S) \
  run_training.pbs
```

```bash
python scripts/run_training.py \
  --config "$WORK_CONFIG_PATH" \
  --fold-id "$fold_id" \
  --run-stem "$RUN_STEM"
```

The current default config uses token-wise chunked supervision:

```yaml
data:
  sequence_window: 256
model:
  local_attention_context_tokens: 128
training:
  batch_size: 32
  gradient_accumulation_steps: 1
  eval_batch_size: 128
  preload_data_to_memory: true
  sequence_supervision:
    mode: token_chunk
    loss_warmup_tokens: 128
    chunk_stride: 128
    neutral_weighting: loss_weight
```

With this setup, each chunk contains 256 events, the first 128 positions are
context warmup, and the loss/evaluation are computed on the remaining 128
positions. `gradient_accumulation_steps: 2` would accumulate two micro-batches
before one optimizer update, giving an effective optimizer batch of about 64
chunks without holding both micro-batches' activations in memory at once. The
neutral-to-directional ratio is applied as deterministic token-level loss
weighting fitted once on the train set: directional labels have weight `1.0`,
while neutral labels receive the same expected contribution that neutral
downsampling would have produced.
The dataloader still iterates over chunks and no neutral token is randomly
dropped from the loss.

`preload_data_to_memory: true` can require substantial host RAM for large folds.
The current PBS script requests fewer CPUs than older runs but keeps high memory
because the compact `.npy` shards may be loaded into RAM:

```bash
#PBS -l select=1:ncpus=8:mem=128gb:ngpus=1
```

`run_training.pbs` saves the PBS stdout/stderr stream under:

```text
logs/<RUN_STEM>/pbs/run_logs.txt
```

and mirrors that stream under:

```text
$HOME/run_outputs/<RUN_STEM>/run_logs.txt
```

To resume after walltime, reuse the same `TRAINING_RUN_STEM` and pass resume
arguments through the PBS variable. This works for classification and
action-value regression:

```bash
qsub -v TRAINING_RUN_STEM="$RUN_STEM",TRAINING_RESUME_ARGS=--resume-latest run_training.pbs
```

For action-value regression, `training_state_latest.pth` is written after every
validated epoch and resumes at the next epoch. Keep the same config, objective,
quantiles, gradient accumulation, supervision mode, and `TRAINING_RUN_STEM`.

For W&B online tracking from PBS, pass the API key through the environment, for
example:

```bash
qsub -v WANDB_API_KEY,WANDB_MODE=online,TRAINING_RUN_STEM="$RUN_STEM" run_training.pbs
```

If the compute node has no network access, run offline and sync later:

```bash
qsub -v WANDB_MODE=offline,TRAINING_RUN_STEM="$RUN_STEM" run_training.pbs
wandb sync --sync-all logs/<RUN_STEM>
```

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

An interrupted array run can be resumed in the same way:

```bash
qsub -J 1-$N_FOLDS \
  -v TRAINING_RUN_STEM="$RUN_STEM",TRAINING_RESUME_ARGS=--resume-latest \
  run_training_folds_array.pbs
```

For the prebuilt FI2010/TLOB configuration, preprocessing is skipped and the
matching config/folds files must be selected together:

```bash
N_FOLDS=$(awk 'NF && $1 !~ /^#/ { c++ } END { print c + 0 }' configs/folds_fi2010.txt)
RUN_STEM=fi2010_tlob_h100_$(date +%Y%m%d_%H%M%S)
qsub -J 1-$N_FOLDS \
  -v TRAINING_CONFIG=configs/config_TLOB_F1_2010.yaml,FOLDS_FILE=configs/folds_fi2010.txt,TRAINING_RUN_STEM="$RUN_STEM" \
  run_training_folds_array.pbs
```

The alternative `config_TLOB_F1_2010_2.yaml` uses
`FOLDS_FILE=configs/folds_fi2010_2.txt` instead.

For `run_training_folds_array.pbs`, each fold task writes its live PBS stream to:

```text
logs/<RUN_STEM>/<fold_id>/run_logs.txt
```

### 6. Evaluate

After training, evaluate a selected checkpoint with `scripts/evaluate_model.py`.
The script can use the same compact sequence shards, supports token-chunk
inference, and can write validation/test probabilities for downstream
calibration, threshold fitting, PR curves, and backtests.

Typical evaluation is:

```bash
PYTHONPATH="$PWD/src:${PYTHONPATH:-}" \
python scripts/evaluate_model.py \
  --config results/<RUN_STEM>/<fold_id>/config.yaml \
  --checkpoint results/<RUN_STEM>/<fold_id>/best_lob_transformer.pth \
  --sequence-dir data/sequences/<fold_id> \
  --split test \
  --output-dir results/<RUN_STEM>/<fold_id>/test_eval \
  --save-probabilities
```

Directional thresholds should be fit on validation probabilities and then
applied to test probabilities, so test metrics remain out-of-sample.

## Status

This repository is under active development as part of my thesis. The current
implementation provides the core preprocessing pipeline, economically motivated
labels, token-wise chunked training, checkpoint/resume support, optional W&B
tracking, directional thresholding, and a dual-attention transformer case study.

The main ongoing work is experimental and evaluative: I am refining the label
framework and validating how fixed-rate directional metrics behave on validation
and test splits, and comparing model behaviour across alternative LOB
representations. I intend to perform some ablation studies on the case study model, as well as comparing it to several baselines models, to evaluate the hypotheses on kinematic streams and MoE usefulness. 
