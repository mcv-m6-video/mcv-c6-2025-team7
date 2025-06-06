#!/bin/bash
#SBATCH --job-name=ft_reg8
#SBATCH --output=/home/xavier/Projects/MasterCV/C6_Lab/Week7/Exp2_TCN_LSTM_Regnet8/log_out/run_job_%j.out
#SBATCH --error=/home/xavier/Projects/MasterCV/C6_Lab/Week7/Exp2_TCN_LSTM_Regnet8/log_err/run_job_%j.err
#SBATCH --nice=100
#SBATCH --partition=sbatch

# ensure log directories exist
mkdir -p /home/xavier/Projects/MasterCV/C6_Lab/Week7/Exp2_TCN_LSTM_Regnet8/log_out
mkdir -p /home/xavier/Projects/MasterCV/C6_Lab/Week7/Exp2_TCN_LSTM_Regnet8/log_err

# activate your virtualenv
source /home/xavier/Projects/MasterCV/C6_Lab/.soccerenv/bin/activate
export PYTHONUNBUFFERED=1

# run your script (use full path)
srun -u python main_spotting_TCN_r008.py --model TCN_r8 --use_tpn --optuna
