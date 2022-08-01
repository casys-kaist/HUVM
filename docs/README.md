# HUVM - Memory Harvesting in Multi-GPU Systems with Hierarchical Unified Virtual Memory (USENIX ATC 2022)

## Table of Contents

- [1. System requirements (Tested environment)](#1-system-requirements-tested-environment)
  - [1.1. Hardware requirements](#11-hardware-requirements)
  - [1.2. Software requirements](#12-software-requirements)
- [2. Dependent package installation](#2-dependent-package-installation)
- [3. Download source code](#3-download-source-code)
- [4. Setup benchmarks](#4-setup-benchmarks)
  - [4.1. cuGraph](#41-cugraph)
  - [4.2. pytorch](#42-pytorch)
- [5. Install HUVM](#5-install-huvm)
- [6. Run benchmarks](#6-run-benchmarks)


## Publication
- Paper: https://www.usenix.org/conference/atc22/presentation/choi-sangjin
- Authors: Sangjin Choi<sup>1</sup>, Taeksoo Kim<sup>1</sup>, Jinwoo Jeong, Rachata Ausavarungnirun, Myeongjae Jeon, Youngjin Kwon, and Jeongseob Ahn

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

- CUDA Toolkit 11.0

This is not a requirement because we use docker for running benchmarks but if you want to run applications outside of docker install CUDA Toolkit from the [nvidia website](https://developer.nvidia.com/cuda-11.0-download-archive).

- NVIDIA GPU driver 460.67

HUVM is built on top of NVIDIA GPU driver version 460.67. Since the default driver version included in CUDA Toolkit is not 460.67, we should install the driver with an additional installer from the [nvidia website](https://www.nvidia.com/Download/driverResults.aspx/171392/en-us).

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

Make sure you include shm-size option. Note that loop option is just for deciding how long the application runs. n_workers option implies the number of GPUs which starts from visible_devices. For instance, if n_workers is 2 and visible_devices is 1,2,3, the application runs in GPU1 and GPU2 and it can harvest GPU3. Note that visible_devices option will be sorted in the cugraph library so we cannot specifically designate a GPU to an application with in the context of HUVM becase harvestable GPU should be visible and be included in visible_devices.  

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

HUVM is developed on top of ```nvidia-uvm``` kernel module. We need to swap the default ```nvidia-uvm``` module which is installed by the NVIDIA GPU driver installer with our HUVM module. Our ```load_driver.sh``` script compiles the kernel module and swaps the ```nvidia-uvm``` module. For more detailed usage of the ```load_driver.sh``` script, refer to [README-parameters](https://github.com/casys-kaist/HUVM/blob/main/docs/README-parameters.md).

```shell
$ cd HUVM/scripts
$ ./load_driver.sh
```

## 6. Run benchmarks

For all bash scripts, we should setup ```project_home``` variable which is the absolute directory of ```HUVM``` (e.g. ```/home/ubuntu/HUVM```). Run the script in the ```HUVM/scripts/fig#``` directory and the result will be saved in the same directory starting with a date. The first run of the script will be a warmup run. You can see how the memory gets harvested using the ```nvidia-smi``` command. 
Compare the results in our paper with the csv file.

#### 6.1 Figure 6: Execution time speedup on four harvesting scenarios

We evaluate the execution time of multiple workloads in four scenarios by varying the type of jobs and number of harvesters. Check out the paper for details about the scenario. There are 5 scripts in the fig6 directory. (case1_pagerank.sh, case2_bfs.sh, case3_wcc.sh, case4_louvain.sh and case4_wcc.sh)
For case 4, the runtime of louvain and wcc is similar. Because we cannot know for sure which application will end first, I have made two scripts that ends when one harvestor ends. The two scripts in the same case is exactly running as a same way but it will save a single csv file each that measures the runtime of a single harvestor application.

```shell
$ cd HUVM/scripts/fig6
$ ./case1_pagerank.sh
$ cd [CURRENT_DATE]_case1
$ cat pagerank.csv
```

#### 6.2. Figure 7: Effectiveness of individual techniques

We decompose the contribution of the performance improvement into individual schemes. (H, H+PE, H+PE+LP, H+PE+LP+PLF, H+PE+LP+PLF+LPF, H+PE+LP+PLF+MPF)

```shell
$ cd HUVM/scripts/fig7
$ ./case1_pagerank.sh
$ cd [CURRENT_DATE]_case1
$ cat pagerank.csv
```

#### 6.3. Figure 8: Sensitivity to the amount of prefetch

We evaluate the sensitivity study for our next line and stride prefetches used in multi-path parallel prefetcher. We varied the prefetch amount from 2MB to 32MB with a 2MB stride. 

```shell
$ cd HUVM/scripts/fig8
$ ./case1_pagerank.sh
$ cd [CURRENT_DATE]_case1
$ cat pagerank.csv
```

#### 6.4. Figure 9: Sensitivity to the size of spare memory

We simulated a scenario with 2 GPUs by manually varying the amount of spare memory from 5% to 60%. One GPU is running memory-intensive graph analytic workload and the other GPU is yileding spare memory with an appropriate size of ```cudaMalloc```. 

```shell
$ cd HUVM/scripts/fig9
$ ./pagerank.sh
$ cd [CURRENT_DATE]_pagerank
$ cat pagerank.csv
```
