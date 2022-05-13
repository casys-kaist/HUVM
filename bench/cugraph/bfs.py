from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
import cugraph.comms as Comms
import cugraph.dask as dask_cugraph
import cugraph
import dask_cudf
import time
import urllib.request
import os
import sys
import rmm
import argparse

def main():
    parser=argparse.ArgumentParser()
    description='''python bfs.py --n_workers 1 --visible_devices 0,1,2,3
                    --dataset /HUVM/dataset/graph/soc-twitter-2010.csv --loop'''
    parser.add_argument('--n_workers', type=int, required=True, help='number of workers')
    parser.add_argument('--visible_devices', type=str, required=True,
                    help='comma seperated CUDA_VISIBLE_DEVICES (e.g. 0,1,2,3)')
    parser.add_argument('--dataset', type=str, required=True, help='path to graph dataset')
    parser.add_argument('--loop', default=False, action='store_true', help='run one time or in loop')
    args=parser.parse_args()

    cluster = LocalCUDACluster(rmm_managed_memory=True, rmm_pool_size="50GB", 
                                CUDA_VISIBLE_DEVICES=args.visible_devices, n_workers=args.n_workers)
    client = Client(cluster)
    Comms.initialize(p2p=True)
    assert(rmm.is_initialized())

    # Helper function to set the reader chunk size to automatically get one partition per GPU  
    chunksize = dask_cugraph.get_chunksize(args.dataset)

    # Multi-GPU CSV reader
    e_list = dask_cudf.read_csv(args.dataset, chunksize = chunksize, delimiter=' ', 
                                names=['src', 'dst'], dtype=['int32', 'int32'])

    # Create a directed graph using the source (src) and destination (dst) vertex pairs from the Dataframe 
    G = cugraph.DiGraph()
    G.from_dask_cudf_edgelist(e_list, source='src', destination='dst')

    if args.loop:
        while True:
            t_start = time.time()
            df = dask_cugraph.bfs(G, 1)
            print("Out: ", time.time()-t_start)
    else:
        t_start = time.time()
        df = dask_cugraph.bfs(G, 1)
        print("Out: ", time.time()-t_start)

    Comms.destroy()
    client.close()
    cluster.close()

if __name__ == "__main__":
    main()
