#/bin/bash
project_home="/home/ubuntu/HUVM"
free_port=7777
current_date=$(date '+%Y-%m-%d_%H:%M:%S')
output_path="${current_date}_case4"
output_raw="${output_path}/raw.txt"
output_filename="louvain"
harvestor_command_1="docker run --gpus all -d --rm -it --name huvm_harvestor_1 -v ${project_home}:/HUVM sjchoi/huvm:init python /HUVM/bench/cugraph/wcc.py --n_workers 1 --visible_devices 0,1,2,3 --dataset /HUVM/dataset/graph/soc-twitter-2010.csv --loop"
harvestor_command_2="docker run --gpus all --rm -it --name huvm_harvestor_2 -v ${project_home}:/HUVM sjchoi/huvm:init python /HUVM/bench/cugraph/louvain.py --n_workers 2 --visible_devices 1,2,3 --dataset /HUVM/dataset/graph/web-uk-2005-all.mtx"
harvestee_command="docker run --gpus all --rm -d -it --name huvm_harvestee -e CUDA_VISIBLE_DEVICES='3' -v ${project_home}:/HUVM sjchoi/huvm:init python /HUVM/bench/pytorch/main.py -a resnet101 -b 64 --dist-url 'tcp://127.0.0.1:${free_port}' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /HUVM/dataset/imagenet"

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
    eval ${harvestee_command} 
    sleep 15
    eval ${harvestor_command_1}
    local output=`${harvestor_command_2} 2>&1 | grep "Out:"`
    echo "${output_filename}/${output_header}" >> "${output_raw}"
    echo "${output}" >> "${output_raw}"
    echo >> "${output_raw}"
    output=`echo "${output:6:-1}"`
    add_col_to_csv "${output_path}/${output_filename}.csv" "${output_header}" "${output}"
    docker container logs huvm_harvestor_1 > ${output_path}/wcc_raw.txt
    docker kill huvm_harvestor_1
    docker container logs huvm_harvestee > ${output_path}/resnet_raw.txt
    docker kill huvm_harvestee
    output=` grep -o 'Time[^;]*Data' ${output_path}/resnet_raw.txt | tail -n 1`
    output=`echo "${output:14:-6}"`
    add_col_to_csv "${output_path}/resnet.csv" "${output_header}" "${output}"
    pkill python
    rm ${output_path}/*_raw.txt
    rm ${output_path}/raw.txt
    sleep 10
}

echo "Figure 6: Case-4 louvain"

output_header="Warmup"
${project_home}/scripts/load_driver.sh -p "uvm_hierarchical_memory=1 uvm_cpu_large_page_support=1 uvm_reserve_chunk_enable=1 uvm_parallel_fault_enable=1 uvm_prefetch_flags=3 uvm_prefetch_num_chunk=16 uvm_prefetch_stride=2"
run_benchmark

output_header="Base"
${project_home}/scripts/load_driver.sh -b
run_benchmark

output_header="pre-ef-host"
${project_home}/scripts/load_driver.sh -p "uvm_hierarchical_memory=0 uvm_cpu_large_page_support=0 uvm_reserve_chunk_enable=1 uvm_parallel_fault_enable=0 uvm_prefetch_flags=1 uvm_prefetch_num_chunk=16 uvm_prefetch_stride=2"
run_benchmark

output_header="Ours"
${project_home}/scripts/load_driver.sh -p "uvm_hierarchical_memory=1 uvm_cpu_large_page_support=1 uvm_reserve_chunk_enable=1 uvm_parallel_fault_enable=1 uvm_prefetch_flags=3 uvm_prefetch_num_chunk=16 uvm_prefetch_stride=2"
run_benchmark

