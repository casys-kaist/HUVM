#/bin/bash
project_home="/home/ubuntu/HUVM"
free_port=7777
current_date=$(date '+%Y-%m-%d_%H:%M:%S')
output_path="${current_date}_case4"
output_raw="${output_path}/raw.txt"
output_filename="resnet"
harvestor_command="CUDA_VISIBLE_DEVICES=1,2 ${project_home}/bench/microbench/vectorAdd/vectorAdd"
harvestee_command="docker run --gpus all --rm -d -it --name huvm_harvestee -e CUDA_VISIBLE_DEVICES='2' -v ${project_home}:/HUVM sjchoi/huvm:init python /HUVM/bench/pytorch/main.py -a resnet101 -b 64 --dist-url 'tcp://127.0.0.1:${free_port}' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /HUVM/dataset/imagenet"

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
    sleep 30
    eval ${harvestor_command} ${overcommit} ${threads} ${num_iter}
    docker logs huvm_harvestee > ${output_path}/${output_header}_resnet_raw.txt
    docker kill huvm_harvestee
    output=`grep -o 'Time[^;]*Data' ${output_path}/${output_header}_resnet_raw.txt | tail -n 1`
    output=`echo "${output:14:-6}"`
    add_col_to_csv "${output_path}/${output_filename}.csv" "${output_header}" "${output}"
    sleep 10
}

echo "Interference"

output_header="Warmup"
overcommit="L"
threads="16"
num_iter="20"
${project_home}/scripts/load_driver.sh
run_benchmark

output_header="Low_16"
overcommit="L"
threads="16"
${project_home}/scripts/load_driver.sh 
run_benchmark

output_header="Low_32"
overcommit="L"
threads="32"
${project_home}/scripts/load_driver.sh 
run_benchmark

output_header="Low_64"
overcommit="L"
threads="64"
${project_home}/scripts/load_driver.sh 
run_benchmark

output_header="Mid_16"
overcommit="M"
threads="16"
${project_home}/scripts/load_driver.sh 
run_benchmark

output_header="Mid_32"
overcommit="M"
threads="32"
${project_home}/scripts/load_driver.sh 
run_benchmark

output_header="Mid_64"
overcommit="M"
threads="64"
${project_home}/scripts/load_driver.sh 
run_benchmark

output_header="High_16"
overcommit="H"
threads="16"
${project_home}/scripts/load_driver.sh 
run_benchmark

output_header="High_64"
overcommit="H"
threads="32"
${project_home}/scripts/load_driver.sh 
run_benchmark

output_header="High_64"
overcommit="H"
threads="64"
${project_home}/scripts/load_driver.sh 
run_benchmark

