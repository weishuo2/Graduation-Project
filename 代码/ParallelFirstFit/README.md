# Parallel Graph coloring implementation in C++

##### Algorithm
This is a CUDA implementation of the topology driven Parallel Graph Coloring Algorithm described in **Pingfan Li et al., High Performance Parallel Graph Coloring on GPGPUs, IPDPSW, 2016**

* We read in a MatrixMarket graph file to internal data structures for a Compressed Sparse Row representation (See `src/include/CSRGraph.h`)

* We're going with a topology based approach here (rather than worklist based). It should be noted that the worklist based approach will definitely give better performance. With this topology based apporach, we expect the kernel to suffer from a lot of warp divergence and load imbalance and perform poorly as a result.

* An inherent issue with the CSR storage is the unaligned and uncoalesced memory access pattern. See this [GTC 2010 Talk on the cuSparse Library from Nvidia](http://on-demand.gputechconf.com/gtc/2010/presentations/S12070-Cusparse-Library-a-Set-of-Basic-Linear-Algebra-Subroutines-for-Sparse-Matrices.pdf). Need to explore additional storage formats.

* Main algorithm:
There are three main parts to this:
1. Initialize - Initialize the "book keeping" datastructures
2. Actual coloring algorithm:
* We'll assign each thread one vertex to deal with as described below - hence the nomenclature topology driven
  * Each Thread:
    * Figure out which vertex you're to process based on the threadID
    * Examine the vertex's neighbors and find the first available color thats not disallowed and assign that.
3. Resolve incorrect coloring from Step-2. 

Repeat 2, 3 till the graph is colored.

## Usage
1. Build the GraphReader library. This is a library written to read in MTX (Matrix Market) graph ASCII files. It'll be dynamically linked to our executable. See the `README.md` in `src/GraphReader` for details.
```
$ cd ../src/GraphReader && make
```
2. Build this program:
   All built components will be in a new folder `build/`. 
   This build uses NVCC's separate compile and linking features as described in a [Parallel Forall post here.](https://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code/). See `Makefile.common` for its internals.
   Set `CUDA\_LIB\_DIR` to point to your local CUDA library directory. Default: `/usr/local/cuda/lib64`
   Set `SM_ARCH` for your GPU. Default: `sm_30`
```
$ make 
```
3. Run the program: The coloring info for the given graph will be written to STDOUT and an output text file.
```
$ ./build/ParallelFirstFit <input-mtx-file> <Output-txt-file>
```
_TODO: Add instructions on how to run the self-test script_

