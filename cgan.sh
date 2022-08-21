#!/bin/bash 
#SBATCH -J Cgan
#SBATCH -t 5760:00
#SBATCH --gres=gpu:V100:1
#SBATCH -p dell
#SBATCH -c 10
#SBATCH -o log/new-experiments/baselines/MyGAN-pure_multiseed.out
#SBATCH -e log/FinEvent_train_9.err

source /home/LAB/luogx/anaconda3/etc/profile.d/conda.sh
conda activate lgx
#python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator origin --n_epochs 20 --to1_epochs 20
#python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator cat --n_epochs 20 --to1_epochs 20
#python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator gru_2 --n_epochs 20 --to1_epochs 20
#python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator mean --n_epochs 20 --to1_epochs 20
#python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator gru --n_epochs 20 --to1_epochs 20

python /home/LAB/luogx/lcy/viewModel/myGAN-pure.py --dataset ppmi_622 --generator origin --n_epochs 20 --to1_epochs 20
python /home/LAB/luogx/lcy/viewModel/myGAN-pure.py --dataset ppmi_622 --generator origin --n_epochs 20 --to1_epochs 20
python /home/LAB/luogx/lcy/viewModel/myGAN-pure.py --dataset ppmi_622 --generator origin --n_epochs 20 --to1_epochs 20
python /home/LAB/luogx/lcy/viewModel/myGAN-pure.py --dataset ppmi_622 --generator origin --n_epochs 20 --to1_epochs 20
python /home/LAB/luogx/lcy/viewModel/myGAN-pure.py --dataset ppmi_622 --generator origin --n_epochs 20 --to1_epochs 20
python /home/LAB/luogx/lcy/viewModel/myGAN-pure.py --dataset ppmi_622 --generator origin --n_epochs 20 --to1_epochs 20
python /home/LAB/luogx/lcy/viewModel/myGAN-pure.py --dataset ppmi_622 --generator origin --n_epochs 20 --to1_epochs 20
python /home/LAB/luogx/lcy/viewModel/myGAN-pure.py --dataset ppmi_622 --generator origin --n_epochs 20 --to1_epochs 20
python /home/LAB/luogx/lcy/viewModel/myGAN-pure.py --dataset ppmi_622 --generator origin --n_epochs 20 --to1_epochs 20
python /home/LAB/luogx/lcy/viewModel/myGAN-pure.py --dataset ppmi_622 --generator origin --n_epochs 20 --to1_epochs 20
