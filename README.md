### Pytorch Multi GPU Cluster Example

## SLURM Interactive Mode Usage

* srun -p gpu-titanxp -N 1 -n 4 -t 02:00:00 --gres=gpu:4 --pty /bin/bash -l
gpu-titanxp 리소스 자원 중에서, Node 1대의 CPU 4개와 GPU 4장을 사용하겠다는 의미입니다. 최대 작업 시간은 2시간입니다.

* srun -p gpu-2080ti -N 1 -n 4 -t 02:00:00 --gres=gpu:4 --pty /bin/bash -l
