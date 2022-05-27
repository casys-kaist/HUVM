#!/bin/bash
slack "fig 9 start"
./pagerank.sh
slack "pagerank fin"
./bfs.sh
slack "bfs fin"
./wcc.sh
slack "wcc fin"
./louvain.sh
slack "louvain fin"
