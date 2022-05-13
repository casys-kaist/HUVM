#/bin/bash
project_home="/home/ubuntu/HUVM"
current_date=$(date '+%Y-%m-%d_%H:%M:%S')
output_path="${current_date}_louvain"
output_raw="${output_path}/raw.txt"
output_filename="louvain"
harvestor_command="docker run --gpus all --rm -it --name huvm_harvestor -v ${project_home}:/HUVM sjchoi/huvm:init python /HUVM/bench/cugraph/louvain.py --n_workers 1 --visible_devices 1,2 --dataset /HUVM/dataset/graph/web-wikipedia_link_en13-all.edges"
harvestee_command="./default_tasks.sh"

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
    ${harvestee_command} ${mem_occupy} &
    sleep 10
    local output=`${harvestor_command} 2>&1 | grep "Out:"`
    echo "${output_filename}/${output_header}" >> "${output_raw}"
    echo "${output}" >> "${output_raw}"
    echo >> "${output_raw}"
    output=`echo "${output:6:-1}"`
    add_col_to_csv "${output_path}/${output_filename}.csv" "${output_header}" "${output}"
    pkill default_tasks
    sleep 10
}

echo "Figure 9: Louvain"

output_header="Warmup"
mem_occupy=1000
${project_home}/scripts/load_driver.sh
run_benchmark

output_header="Base"
mem_occupy=1000
${project_home}/scripts/load_driver.sh -b
run_benchmark

output_header="5%"
mem_occupy=15052
${project_home}/scripts/load_driver.sh
run_benchmark

output_header="10%"
mem_occupy=14244
${project_home}/scripts/load_driver.sh
run_benchmark

output_header="20%"
mem_occupy=12628
${project_home}/scripts/load_driver.sh
run_benchmark

output_header="40%"
mem_occupy=9396
${project_home}/scripts/load_driver.sh
run_benchmark

output_header="60%"
mem_occupy=6164
${project_home}/scripts/load_driver.sh
run_benchmark

