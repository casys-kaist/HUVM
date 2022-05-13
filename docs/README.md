# HUVM - Memory Harvesting in Multi-GPU Systems with Hierarchical Unified Virtual Memory 

- [1. System requirements (Tested environment)](#1-system-requirements-tested-environment)
  - [1.1. Hardware requirements](#11-hardware-requirements)
    - [1.1.1. Host machine](#111-host-machine)
    - [1.1.2. NVLink topology](#112-nvlink-topology)
  - [1.2. Software requirements](#12-software-requirements)
- [2. Dependent package installation](#2-dependent-package-installation)
- [3. Download source code](#3-download-source-code)
- [4. Setup benchmarks](#4-setup-benchmarks)
  - [4.1. cuGraph](#41-cugraph)
  - [4.2. pytorch](#42-pytorch)
- [5. Install HUVM](#5-install-huvm)
- [6. Run benchmarks](#6-run-benchmarks)
  - [6.1. Inter-job harvesting](#61-inter-job-harvesting)
  - [6.2. Intra-job harvesting](#62-intra-job-harvesting)

## 1. System requirements (Tested environment)

### 1.1. Hardware requirements

#### 1.1.1. Host machine

- AWS P3.8xlarge instance (NVIDIA Deep Learning  AMI v21.06.0-46a68101-e56b-41cd-8e32-631ac6e5d02b)
- GPU: NVIDIA V100 (16GB) x 4ea
- Memory: 244GB DDR4 DRAM
- CPU: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
- NVLink 2.0
- PCIe 3.0

#### 1.1.2. NVLink topology 

All gpus should be connected with NVLink with each other.
If there are more than 2 GPUs, this topology is only possible for DGX-based machines. 

```shell
$ nvidia-smi topo -m 
	GPU0	GPU1	GPU2	GPU3	CPU Affinity	NUMA Affinity
GPU0	 X 	NV1	NV1	NV2	0-31		N/A
GPU1	NV1	 X 	NV2	NV1	0-31		N/A
GPU2	NV1	NV2	 X 	NV2	0-31		N/A
GPU3	NV2	NV1	NV2	 X 	0-31		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

### 1.2. Software requirements 

- Operating system: Ubuntu 20.04
- Kernel version: 5.8.0-1041-aws
- GPU driver: NVIDIA GPU driver version 460.67
- Applications: cuGraph (version 21.12), Pytorch (version 1.10.1)
- Dataset: soc-twitter-2010, web-uk-2005, soc-twitter-2010, soc-sinaweibo, web-wikipedia, ImageNet

## 2. Dependent package installation

- build-essential

```shell
$ sudo apt update
$ sudo apt install build-essential
```

- CUDA Toolkit 11.2

This is not a requirement because we use docker for running benchmarks but if you want to run applications outside of docker install CUDA Toolkit from the [nvidia website](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal).

```shell
$ wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run
$ sudo sh cuda_11.2.0_460.27.04_linux.run
```

- NVIDIA GPU driver 460.67

HUVM is built on top of NVIDIA GPU driver version 460.67. Since the default driver version included in CUDA Toolkit 11.2 is not 460.67, we should install the driver with an additional installer from the [nvidia website](https://www.nvidia.com/Download/driverResults.aspx/171392/en-us).

```shell
$ wget https://us.download.nvidia.com/XFree86/Linux-x86_64/460.67/NVIDIA-Linux-x86_64-460.67.run
$ sudo sh NVIDIA-Linux-x86_64-460.67.run
```

## 3. Download source code

```shell
$ git clone git@github.com:casys-kaist/HUVM.git
$ cd HUVM
```

## 4. Setup benchmarks

### 4.1. cuGraph

- Download graph dataset

```shell
$ sudo apt install unzip
$ cd HUVM/scripts
$ ./download_graph_dataset.sh
```

After running the script, you will find graph dataset files (soc-sinaweibo.mtx, soc-twitter-2010.csv, web-ClueWeb09-50m.edges, web-uk-2005-all.mtx, web-cc12-PayLevelDomain.edges, web-wikipedia_link_en13-all.edges) in ```HUVM/dataset```.

- Run docker 

Make sure you include shm-size option. Note that harvestor option is just for deciding how long the application runs (harvestor runs 1 time and harvestee runs in infinite loop).

```
$ docker run --gpus all --shm-size=1g -it -v ~/HUVM:/HUVM sjchoi/huvm:init bash

(rapids) root@ebd47677ba5a:~# cd /HUVM/bench/cugraph
(rapids) root@ebd47677ba5a:~# python pagerank.py -h
usage: pagerank.py [-h] --n_workers N_WORKERS --visible_devices VISIBLE_DEVICES --dataset DATASET [--harvestor]

optional arguments:
  -h, --help            show this help message and exit
  --n_workers N_WORKERS
                        number of workers
  --visible_devices VISIBLE_DEVICES
                        comma seperated CUDA_VISIBLE_DEVICES (e.g. 0,1,2,3)
  --dataset DATASET     path to graph dataset
  --harvestor           harvestor or harvestee
(rapids) root@ebd47677ba5a:~# python pagerank.py --n_workers 2 --visible_devices 1,2 --dataset /HUVM/dataset/soc-twitter-2010.csv --harvestor
distributed.preloading - INFO - Import preload module: dask_cuda.initialize
distributed.preloading - INFO - Import preload module: dask_cuda.initialize
Out:  63.40642762184143
```

### 4.2. Pytorch

- Download imagenet dataset 

Download ImageNet dataset from the [imagenet website](https://www.image-net.org/). Locate the file in ```HUVM/dataset/imagenet``` and preprocess it. 

- Run docker

```
$ docker run --gpus all --shm-size=1g -it -v ~/HUVM:/HUVM sjchoi/huvm:init bash

(rapids) root@6549bfdafe0c:~# cd /HUVM/bench/pytorch
(rapids) root@6549bfdafe0c:/HUVM/bench/pytorch# CUDA_VISIBLE_DEVICES=1,2 python3 main.py -a vgg16 -b 128 --dist-url 'tcp://127.0.0.1:FREE_PORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /HUVM/dataset/imagenet
```

## 5. Install HUVM

HUVM is developed on top of ```nvidia-uvm``` kernel module. We need to swap the default ```nvidia-uvm``` module which is installed by the NVIDIA GPU driver installer with our HUVM module. Our ```load_driver.sh``` script compiles the kernel module and swaps the ```nvidia-uvm``` module. For more detailed usage of the ```load_driver.sh``` script, refer to [README-parameters](https://github.com/sjchoi1/huvm_tmp/blob/main/docs/README-parameters.md).

```shell
$ cd HUVM/scripts
$ ./load_driver.sh
```

## 6. Run benchmarks

For all bash scripts, we should setup ```project_home``` variable which is the absolute directory where the ```HUVM``` directory is cloned. (e.g. ```/home/ubuntu/HUVM```). Run the script in the ```HUVM/scripts/fig#``` directory and the result will be saved in the same directory with the bash script. 

### 6.1. Inter-job harvesting

#### 6.1.1. Figure 6: Execution time speedup on four harvesting scenarios



#### 6.1.2. Figure 7: Effectiveness of individual techniques 

#### 6.1.3. Figure 8: Sensitivity to the amount of prefetch

#### 6.1.4. Figure 9: Sensitivity to the size of spare memory 

### 6.2. Intra-job harvesting

#### 6.2.1. Figure 10: Throughput improvement for single training workloads


