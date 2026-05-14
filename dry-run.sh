#!/bin/bash
#PBS -N dry-run
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=8:mem=24gb:ngpus=1:gpu_type=A100

eval "$(~/miniforge3/bin/conda shell.bash hook)”
conda activate thesis

cd $PBS_O_WORKDIR
cp -r $PBS_O_WORKDIR/thesis/ $TMPDIR

python $TMPDIR/thesis/scripts/vram_dry_run.py \
  --batch-size 32 \
  --sequence-length 100 \
  --num-features 240 \
  --steps 5 \
> $TMPDIR/dry_run_results.txt

cp $TMPDIR/dry_run_results.txt $PBS_O_WORKDIR/dry_run_results.txt

