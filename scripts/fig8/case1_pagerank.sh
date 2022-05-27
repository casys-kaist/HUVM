#/bin/bash
project_home="/home/ubuntu/HUVM"
free_port=7777
current_date=$(date '+%Y-%m-%d_%H:%M:%S')
output_path="${current_date}_case1"
output_raw="${output_path}/raw.txt"
output_filename="pagerank"
harvestor_command="docker run --gpus all --rm -it --name huvm_harvestor -v ${project_home}:/HUVM sjchoi/huvm:init python /HUVM/bench/cugraph/pagerank.py --n_workers 1 --visible_devices 0,1,2,3 --dataset /HUVM/dataset/graph/soc-twitter-2010.csv"
harvestee_command_1="docker run --gpus all --rm -d -it --name huvm_harvestee_1 -v ${project_home}:/HUVM sjchoi/huvm:init python /HUVM/bench/cugraph/wcc.py --n_workers 1 --visible_devices 3 --dataset /HUVM/dataset/graph/soc-sinaweibo.mtx --loop"
harvestee_command_2="docker run --gpus all --rm -d -it --name huvm_harvestee_2 -e CUDA_VISIBLE_DEVICES='1,2' -v ${project_home}:/HUVM sjchoi/huvm:init python /HUVM/bench/pytorch/main.py -a vgg16 -b 256 --dist-url 'tcp://127.0.0.1:${free_port}' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /HUVM/dataset/imagenet"

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

echo "Figure 8: Pagerank (Case-1)"

output_header="Warmup"
${project_home}/scripts/load_driver.sh
run_benchmark

output_header="H+PE+LP+PLF+MPF(0MB)"
${project_home}/scripts/load_driver.sh -p "uvm_hierarchical_memory=1 uvm_cpu_large_page_support=1 uvm_reserve_chunk_enable=1 uvm_parallel_fault_enable=1 uvm_prefetch_flags=3 uvm_prefetch_num_chunk=0 uvm_prefetch_stride=2"
run_benchmark

output_header="H+LP+PE+PLF+MPF(2MB)"
${project_home}/scripts/load_driver.sh -p "uvm_hierarchical_memory=1 uvm_cpu_large_page_support=1 uvm_reserve_chunk_enable=1 uvm_parallel_fault_enable=1 uvm_prefetch_flags=3 uvm_prefetch_num_chunk=1 uvm_prefetch_stride=2"
run_benchmark

output_header="H+LP+PE+PLF+MPF(4MB)"
${project_home}/scripts/load_driver.sh -p "uvm_hierarchical_memory=1 uvm_cpu_large_page_support=1 uvm_reserve_chunk_enable=1 uvm_parallel_fault_enable=1 uvm_prefetch_flags=3 uvm_prefetch_num_chunk=2 uvm_prefetch_stride=2"
run_benchmark

output_header="H+LP+PE+PLF+MPF(8MB)"
${project_home}/scripts/load_driver.sh -p "uvm_hierarchical_memory=1 uvm_cpu_large_page_support=1 uvm_reserve_chunk_enable=1 uvm_parallel_fault_enable=1 uvm_prefetch_flags=3 uvm_prefetch_num_chunk=4 uvm_prefetch_stride=2"
run_benchmark

output_header="H+LP+PE+PLF+MPF(16MB)"
${project_home}/scripts/load_driver.sh -p "uvm_hierarchical_memory=1 uvm_cpu_large_page_support=1 uvm_reserve_chunk_enable=1 uvm_parallel_fault_enable=1 uvm_prefetch_flags=3 uvm_prefetch_num_chunk=8 uvm_prefetch_stride=2"
run_benchmark

output_header="H+LP+PE+PLF+MPF(32MB)"
${project_home}/scripts/load_driver.sh -p "uvm_hierarchical_memory=1 uvm_cpu_large_page_support=1 uvm_reserve_chunk_enable=1 uvm_parallel_fault_enable=1 uvm_prefetch_flags=3 uvm_prefetch_num_chunk=16 uvm_prefetch_stride=2"
run_benchmark

