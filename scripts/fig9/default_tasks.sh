#!/bin/bash

# ./default_tasks.sh <mem%-to-occupy> [<gpuid-to-active> (default: 1,3)]

trap 'kill $(jobs -p) 2> /dev/null' EXIT

die() {
    echo >&2 -e "${1-}"
    exit 1
}

[[ -z "$1" ]] && die "requires memory percentage to occupy"

mem_occupy="$1"

CUDA_VISIBLE_DEVICES=2 ./manual_occupy.run ${mem_occupy}
