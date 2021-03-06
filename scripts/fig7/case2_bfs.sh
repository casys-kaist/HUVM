#/bin/bash
project_home="/home/ubuntu/HUVM"
free_port_1=7777
free_port_2=7778
current_date=$(date '+%Y-%m-%d_%H:%M:%S')
output_path="${current_date}_case2"
output_raw="${output_path}/raw.txt"
output_filename="bfs"
harvestor_command="docker run --gpus all --rm -it --name huvm_harvestor -v ${project_home}:/HUVM sjchoi/huvm:init python /HUVM/bench/cugraph/bfs.py --n_workers 2 --visible_devices 0,1,2,3 --dataset /HUVM/dataset/graph/web-uk-2005-all.mtx"
harvestee_command_1="docker run --gpus all --rm -d -it --name huvm_harvestee_1 -e CUDA_VISIBLE_DEVICES='2' -v ${project_home}:/HUVM -it sjchoi/huvm:init python /HUVM/bench/pytorch/main.py -a mobilenet_v3_large -b 256 --dist-url 'tcp://127.0.0.1:${free_port_1}' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /HUVM/dataset/imagenet"
harvestee_command_2="docker run --gpus all --rm -d -it --name huvm_harvestee_2 -e CUDA_VISIBLE_DEVICES='3' -v ${project_home}:/HUVM -it sjchoi/huvm:init python /HUVM/bench/pytorch/main.py -a resnet101 -b 64 --dist-url 'tcp://127.0.0.1:${free_port_2}' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /HUVM/dataset/imagenet"

echo ${current_date}
mkdir ${output_path}

# Write given input to new column in csv file. Gets file in $1, header name in
# $2, and input string in $3.
add_col_to_csv() {
    local file=$1
    local header=$2
    local str=$3

    str=`echo -e "${header}\n${str}"`
    
    if [ -f ${file} ]
    then
        str=`paste -d, ${file} <(printf "%s" "${str}")`
    fi
    echo "${str}" > ${file}
}

run_benchmark() {
    echo "start ${output_header}"
    eval ${harvestee_command_1}
    eval ${harvestee_command_2}
    sleep 30
    local output=`${harvestor_command} 2>&1 | grep "Out:"`
    echo "${output_filename}/${output_header}" >> "${output_raw}"
    echo "${output}" >> "${output_raw}"
    echo >> "${output_raw}"
    output=`echo "${output:6:-1}"`
    add_col_to_csv "${output_path}/${output_filename}.csv" "${output_header}" "${output}"
    docker kill huvm_harvestee_1
    docker kill huvm_harvestee_2
    sudo pkill python
    sleep 10
}

echo "Figure 7: BFS (Case-2)"

output_header="Warmup"
${project_home}/scripts/load_driver.sh 
run_benchmark

output_header="Base"
${project_home}/scripts/load_driver.sh -b
run_benchmark

output_header="H"
${project_home}/scripts/load_driver.sh -p "uvm_hierarchical_memory=1 uvm_cpu_large_page_support=0 uvm_reserve_chunk_enable=0 uvm_parallel_fault_enable=0 uvm_prefetch_flags=0 uvm_prefetch_num_chunk=0 uvm_prefetch_stride=2"
run_benchmark

output_header="H+PE"
${project_home}/scripts/load_driver.sh -p "uvm_hierarchical_memory=1 uvm_cpu_large_page_support=0 uvm_reserve_chunk_enable=1 uvm_parallel_fault_enable=0 uvm_prefetch_flags=0 uvm_prefetch_num_chunk=0 uvm_prefetch_stride=2"
run_benchmark

output_header="H+PE+LP"
${project_home}/scripts/load_driver.sh -p "uvm_hierarchical_memory=1 uvm_cpu_large_page_support=1 uvm_reserve_chunk_enable=1 uvm_parallel_fault_enable=0 uvm_prefetch_flags=0 uvm_prefetch_num_chunk=0 uvm_prefetch_stride=2"
run_benchmark

output_header="H+PE+LP+PLF"
${project_home}/scripts/load_driver.sh -p "uvm_hierarchical_memory=1 uvm_cpu_large_page_support=1 uvm_reserve_chunk_enable=1 uvm_parallel_fault_enable=1 uvm_prefetch_flags=0 uvm_prefetch_num_chunk=0 uvm_prefetch_stride=2"
run_benchmark

output_header="H+PE+LP+PLF+LPF"
${project_home}/scripts/load_driver.sh -p "uvm_hierarchical_memory=1 uvm_cpu_large_page_support=1 uvm_reserve_chunk_enable=1 uvm_parallel_fault_enable=1 uvm_prefetch_flags=1 uvm_prefetch_num_chunk=16 uvm_prefetch_stride=2"
run_benchmark

output_header="H+PE+LP+PLF+MPF"
${project_home}/scripts/load_driver.sh -p "uvm_hierarchical_memory=1 uvm_cpu_large_page_support=1 uvm_reserve_chunk_enable=1 uvm_parallel_fault_enable=1 uvm_prefetch_flags=3 uvm_prefetch_num_chunk=16 uvm_prefetch_stride=2"
run_benchmark

