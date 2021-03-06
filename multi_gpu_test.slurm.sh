#!/bin/sh

#SBATCH -J multi_gpu_test
#SBATCH -o multi_gpu_test.%j.out
#SBATCH -p gpu-titanxp
#SBATCH -t 02:00:00

#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --nodelist=n7
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

echo "[ Python Program Start ]"

python3  $HOME/Pytorch-Multi-GPU-Cluster-Example/multi_gpu_test.py

date

squeue  --job  $SLURM_JOBID

echo  "##### END #####"
