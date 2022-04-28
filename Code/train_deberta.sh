#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=hatemoji_deberta

# set number of GPUs
#SBATCH --gres=gpu:8

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=user@email.co.uk

# add module cuda
module load cuda/10.1

# add conda load
module load python/anaconda3

source activate hatemoji

USER_PATH = #`enter path to code directory e.g. ./Hatemoji/Code`
ROUND_ID = #`enter current round e.g. r5,r6,r7`
# Different upsampled train data is loaded from job files so tasks can be run as slurm array
DATA_DIR=`cat $USER_PATH/train_step/jobs/$SLURM_ARRAY_TASK_ID`
echo $USER_PATH
echo $ROUND_ID
echo $DATA_DIR
mkdir -p $USER_PATH/train_step/$ROUND_ID/$DATA_DIR/deberta/

echo "NODELIST HERE"
echo $SLURM_JOB_NODELIST
echo $SLURMD_NODENAME

# Deberta
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python $USER_PATH/run_glue.py \
--model_name_or_path microsoft/deberta-base \
--validation_file $USER_PATH/$ROUND_ID/$DATA_DIR/dev.csv \
--train_file $USER_PATH/train_step/$ROUND_ID/$DATA_DIR/train.csv \
--do_train --do_eval --max_seq_length 512 --learning_rate 2e-5 \
--num_train_epochs 3 --evaluation_strategy epoch \
--load_best_model_at_end --output_dir $USER_PATH/train_step/$ROUND_ID/$DATA_DIR/deberta/ \
--seed 123 \
--cache_dir $USER_PATH/.cache/huggingface/transformers/ \
--overwrite_output_dir > $USER_PATH/train_step/$ROUND_ID/$DATA_DIR/log_train_deberta 2> $USER_PATH/train_step/$ROUND_ID/$DATA_DIR/err_train_deberta

