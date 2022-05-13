# HUVM Kernel Driver

## Installation

```bash
$ cd HUVM/scripts
$ ./load_driver.sh
```

```bash
$ ./load.driver.sh -h
Usage: load_driver.sh [-h] [-d] [-b] [-p <name>=<value>]

Helper for loading nvidia driver.

Available options:

-h, --help      Print this help and exit
-d              Compile with debug configuration
-b              Load stock version of UVM (All params are ignored)
-p              Pass parameters when inserting the module
```

## Module Parameters

`uvm_perf_prefetch_threshold`

Threshold for UVM default prefetcher. It prefetches within the given chunk. Do not change this value when other options are turned on.
(Default: 1)
* Min: 1 (Prefetch all pages in 2MB chunk)
* Max: 100

`uvm_hierarchical_memory`

Enable hierarchical UVM. When enabled, it seeks peer GPUs first for swap space.
(Default: 1)

`uvm_cpu_large_page_support`

Enable CPU large page support. When enabled, it allocate a large page (2 MB) to host.
(Default: 1)

`uvm_reserve_chunk_enable`

Enable reserving chunk (pre-evict).
(Default: 1)

`uvm_reserve_chunk_level`

The aggressiveness level of reserving chunk. 1 level means reserving 10 chunks.
(Default: 5; reserving 50 chunks)

`uvm_parallel_fault_enable`

Enable parallel fault. It fetches page faults to peer GPUs in parallel.
(Default: 1)

`uvm_prefetch_flags`

Indicate prefetch types with flags.
(Default: 3)
* 0: Disable
* 1: Prefetch all to local
* 2: Prefetch all to remote GPU
* 3: Multi-path parallel prefetch

||flags = 1|flags = 2|flags = 3 (# harvestors <= # harvestees)|flags = 3 (# harvestors > # harvestees)|
|------|---|---|------|------|
|resident on CPU|to local|to remote|to remote|to local|
|resident on peer GPU|to local|-|to local|to local|
|resident on local|-|-|-|-|

`uvm_prefetch_num_chunk`

The number of chunks to prefetch. A chunk is 2MB.
(Default: 16; prefetch 32MB)
* Min: 0 (Disable prefetch)
* Max: 64 (128MB)

`uvm_prefetch_stride`

Stride for prefetching.
(Default: 2)

`uvm_debug_counter`

Enable counters for debugging purpose. It counts every situation listed below for every va_space.
(Default: 0)
* EVICTION_TO_HOST
* EVICTION_TO_REMOTE
* PRE_EVICTION_TO_HOST
* PRE_EVICTION_TO_REMOTE
* FETCH_FROM_HOST
* FETCH_FROM_REMOTE
* FETCH_BY_FAULT
* PREFETCH_FROM_HOST
* PREFETCH_FROM_REMOTE
