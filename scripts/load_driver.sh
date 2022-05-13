#!/bin/bash

# NVIDIA DRIVER LOADER
#
# The project tree shows as below:
# huvm
# ├─ driver
# ├─ driver-base
# └─ scripts

set -Euo pipefail

# realpath to get real path when symbolic linked
script_dir=$(cd "$( dirname "$( realpath "${BASH_SOURCE[0]}")")" &>/dev/null && pwd -P)

usage() {
    cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [-h] [-d] [-b] [-p <name>=<value>]

Helper for loading nvidia driver.

Available options:

-h, --help      Print this help and exit
-d              Compile with debug configuration
-b              Load stock version of UVM (All params are ignored)
-p              Pass parameters when inserting the module
EOF
    exit
}

die() {
    local msg="${1-}"
    local code="${2-1}"
    echo >&2 -e "$msg"
    exit "$code"
}

parse_params() {
    # default values of variables set from params
    debug=0
    base_suffix=''
    param=''

    while :; do
        case "${1-}" in
        -h | --help) usage ;;
        -d) debug=1 ;;
        -b) base_suffix="_base" ;;
        -p)
            param="${2-}"
            [[ -z "${param-}" ]] && die "Missing required module parameters"
            shift
            ;;
        -?*) die "Unknown option: $1" ;;
        *) break ;;
        esac
        shift
    done

    return 0
}

parse_params "$@"

driver_dir="${script_dir}/../driver${base_suffix}/"

cd ${driver_dir}
if [ $debug -eq 1 ]; then
    sed -i 's/UVM_BUILD_TYPE = release/UVM_BUILD_TYPE = debug/g' nvidia-uvm/nvidia-uvm.Kbuild
else
    sed -i 's/UVM_BUILD_TYPE = debug/UVM_BUILD_TYPE = release/g' nvidia-uvm/nvidia-uvm.Kbuild
fi

IGNORE_CC_MISMATCH=1 make -j40 > /dev/null
status=$?
if [ $status -ne 0 ]
then
    die "Compilation failed"
fi

sudo nvidia-smi -pm 0 > /dev/null

sudo rmmod nvidia-uvm
sudo insmod nvidia-uvm.ko ${param}

sudo nvidia-smi -pm 1 > /dev/null
