### Pytorch Multi GPU Cluster Example

#### SLURM Interactive Mode Usage

* srun -p gpu-titanxp -N 1 -n 4 -t 02:00:00 --gres=gpu:4 --pty /bin/bash -l
> gpu-titanxp 리소스 자원 중에서, Node 1대의 CPU 4개와 GPU 4장을 최대 2시간동안 사용하겠다는 의미입니다.

* srun -p gpu-2080ti -N 1 -n 4 -t 02:00:00 --gres=gpu:4 --pty /bin/bash -l
> gpu-2080ti 리소스 자원 중에서, Node 1대의 CPU 4개와 GPU 4장을 최대 2시간동안 사용하겠다는 의미입니다.

#### Multi GPU Mode Usage

* 현재 GPU 카드 개수를 출력합니다.
<pre>
python3 check_gpu_count.py
</pre>

* Multi GPU 기능을 테스트합니다.
<pre>
python3 multi_gpu_test.py
</pre>

#### SBATCH Usage

* SBATCH 명령어로 특정한 Job을 실행시킵니다.
<pre>
sbatch check_gpu_count.slurm.sh
sbatch multi_gpu_test.slurm.sh
</pre>

* 실행 중인 Job을 확인하고, 해당 노드로 들어가 그래픽 카드의 사용량을 확인합니다.
<pre>
sinfo
squeue
ssh n13
nvidia-smi
</pre>

* SBATCH를 이용해 Job을 던졌을 때, Multi GPU 연산이 동작하지 않을 수 있습니다.
* 연산이 동작하지 않는 경우에는, 해당 노드에 접속하여 다른 프로세스가 그래픽 카드를 사용 중인지 확인할 수 있습니다.
* 이후에 직접 --nodelist 옵션으로 가동 중이 아닌 노드를 선택하여 Job을 실행할 수 있습니다.
* Multi GPU 연산의 속도가 오히려 더 느린 경우, Batch Size를 조절할 필요가 있습니다. (Batch Size가 커지면 오히려 빨라집니다.)
