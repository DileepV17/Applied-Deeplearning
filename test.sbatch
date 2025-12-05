#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00

source /dss/dsshome1/09/ra47fey3/miniforge3/etc/profile.d/conda.sh
conda activate py39
python src/fine_tuning_clip.py