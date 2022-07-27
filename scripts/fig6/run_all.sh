#!/bin/bash
slack "fig6 start"
./case1_pagerank.sh
slack "case 1 fin"
./case2_bfs.sh
slack "case 2 fin"
./case4_wcc.sh
slack "case 4 fin"
./case4_louvain.sh
slack "case 4 fin"
