#!/bin/bash
slack "fig7 start"
./case1_pagerank.sh
slack "case 1 fin"
./case2_bfs.sh
slack "case 2 fin"
./case3_wcc.sh
slack "case 3 fin"
./case4_louvain.sh
slack "case 4 fin"
