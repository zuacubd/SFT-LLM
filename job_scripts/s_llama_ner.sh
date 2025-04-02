#!/bin/bash
#SBATCH --job-name=Llama_Skill_Extraction
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/s_llama_ner.out
#SBATCH --error=logs/s_llama_ner.err


#source ~/.bashrc
#conda activate honours

PYTHON="/users/40017497/.conda/envs/pumpkin/bin/python"

$PYTHON python_scripts/s_llama_ner.py

