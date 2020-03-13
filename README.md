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
* 자신이 실행 중인 Job이 해당 노드에서 돌아가고 있지 않은 경우, 해당 노드로의 접속이 안 될 수 있습니다.
<pre>
sinfo
squeue
ssh n13
nvidia-smi
</pre>

* SBATCH를 이용해 Job을 던졌을 때, Multi GPU 연산이 동작하지 않을 수 있습니다.
* 연산이 동작하지 않는 경우에는, 해당 노드에 접속하여 다른 프로세스가 그래픽 카드를 사용 중인지 확인할 수 있습니다.
* 이후에 직접 --nodelist 옵션으로 가동 중이 아닌 노드를 선택하여 Job을 실행할 수 있습니다.
* Multi GPU 연산의 속도가 오히려 더 느린 경우, Batch Size를 조절합니다.(Batch Size가 커지면 오히려 빨라집니다.)
* 그렇다고 해서 Batch Size를 너무 크게 만들면, 기존 논문들에 따르면 오히려 (일반화) 성능이 떨어질 수 있습니다.
* 물론 애초에 Batch Size를 크게 하려면 Multi GPU가 필요합니다.
* 데이터셋 자체가 작은 경우 (MNIST 등), Multi GPU가 비효율적으로 동작할 수 있습니다. (CIFAR-10의 경우 정상 동작)
* Multi GPU 모드로 설정해도, 하나의 GPU만 사용되는 경우에는 코드 내부의 오류를 찾아야 합니다. (환경변수 등)
* Multi GPU로  동작하는 [코드](https://github.com/facebookresearch/mixup-cifar10)를 참고합니다.

#### Anoconda Download & Installation

* 아나콘다를 다운로드하고, 설치합니다.
<pre>
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
sh Anaconda3-2019.10-Linux-x86_64.sh
</pre>

* Deactivate 이후에 가상 환경 리스트 확인하기
<pre>
conda deactivate
conda env list 
</pre>

* TensorFlow 전용 가상 환경 구축하기 및 Activate
<pre>
conda create -n tf-cpu-py37 python=3.7
conda activate tf-cpu-py37
</pre>

* 클러스터에서 가상 환경 이용 및 필요한 모듈만 Load
<pre>
ml purge
conda activate tf-cpu-py37
ml load cuda/10.0
ml load cuDNN/cuda/10.0/7.6.4.38
</pre>

* 필요한 라이브러리를 자기자신의 사용자 환경에 설치하기
<pre>
pip install tensorflow-gpu --user
pip install matplotlib --user
pip install image --user
pip install pillow --user
pip install opencv-python --user
pip install dlib --user
</pre>

#### 실제 실험 과정

* 가장 먼저 conda 가상환경을 구축하여, 필요한 라이브러리를 설치합니다.
* 일단 클러스터(Cluster) 접속 이전에 해당 conda 가상환경으로 프로그램을 실행합니다.
* 실행이 잘 되는 경우, srun 명령어로 /bin/bash를 이용하여 특정 클러스터 노드에서 프로그램을 실행합니다.
  * srun -p gpu-titanxp -N 1 -n 1 -t 264:00:00 --gres=gpu:1 --pty /bin/bash -l
* 최종적으로 slurm 파일을 작성한 뒤에 sbatch 명령어로 Job을 실행합니다.
  * sbatch 파일.sh

#### libopenblas 설치 문제

* 파이썬(Python)의 dlib 라이브러리를 사용할 때 다음과 같은 오류가 발생할 수 있습니다.
<pre>
Traceback (most recent call last):
  File "make_adversarial_video.py", line 222, in <module>
    import dlib
ImportError: libopenblas.so.0: cannot open shared object file: No such file or directory
</pre>

* 클러스터 일반 사용자의 경우 sudo 권한이 없어 설치가 불가능합니다.
* 따라서 다음과 같이 소스코드를 직접 받아서 컴파일하여 사용할 수 있습니다.

<pre>
git clone https://github.com/xianyi/OpenBLAS && cd OpenBLAS
CC=gcc make -j$(nproc)
</pre>

* 이후에 특정 세션에서 env | grep LD_LIBRARY_PATH 명령으로 기존의 LD_LIBRARY_PATH 환경변수 확인한 뒤에 내용 복사
* 기존의 해당 환경변수 내용에 :를 붙인 뒤에 /home/dongbinna/additional_library/OpenBLAS 추가 (OpenBLAS 절대 경로)
* 마지막으로 export LD_LIBRARY_PATH='변경된 내용' 으로 환경변수 업데이트 해주기
* 이렇게 해주어도 Illegal instruction (core dumped)와 같은 오류가 발생할 수 있음
  * 저자는 이러한 오류가 발생했었으나, 프로그램이 있는 폴더에서 환경변수 작업 이후에 바로 실행하니 오류가 나오지 않았음

#### 특정 노드에서 무한 반복 스크립트만 실행하는 방법

* 무한 반복 스크립트 (repeat.py)
<pre>
while True:
    pass
</pre>

* Slurm Batch 파일 예제 (bash_batch.sh)
<pre>
#!/bin/sh

#SBATCH -J batch_example
#SBATCH -o batch_example.%j.out
#SBATCH -p gpu-titanxp
#SBATCH -t 264:00:00

#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

python3 repeat.py

date

squeue  --job  $SLURM_JOBID

echo  "##### END #####"
</pre>

* sbatch bash_batch.sh 명령으로 실행
* 이제 해당 Job이 무한히 돌고 있을 때, ssh 명령으로 해당 노드에 접근하여 사용하는 것이 가능하긴 함
* 솔직히 이 방법이 제일 편하더라도, 공용 클러스터 환경이라면 사용하면 문제가 될 수 있음

#### 기타

* 특정 노드에 ssh 명령으로 접속한 뒤에도, pip 명령으로 필요한 패키지를 설치해 바로 쓸 수 있음
* 특정 노드에 ssh 명령으로 접속하는 경우, GPU 자원이 자신의 것만으로 제한되지 않음
  * 그래서 파이썬 프로그램이 다른 사용자의 GPU에 접근하는 경우도 생길 수 있음 (오류는 발생하지 않음)
  * 반면에 다른 사용자에게 점유된 인덱스(Index)가 1인 GPU가 현재 사용 중이지 않아도, 해당 GPU를 사용하도록 처리되지 않아서 메모리(Memory) 오류가 발생할 수 있음
  * 따라서 다음과 같이 특정한 GPU 자원만 이용하도록 코드상에서 제한해 줄 수도 있음
<pre>
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
</pre>
* 별도의 Limit이 걸려있지 않다면, ssh로 접속한 뒤에 학습(Training) 코드를 실행해도 튕기지 않음
