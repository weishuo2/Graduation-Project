#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <thrust/count.h>
#include <numeric>
#include <random>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <helper_cuda.h>
#include "coloring.h"
#include "time.h"
#include "utility.h"

#define MAX_GRID_SIZE 65535

int *same_color_sum;

inline int num_finished(const int n_vertices, 
                 const int * row_ptr, 
                 const int * col, 
                 const int * col_ptr, 
                 const int * row, 
                 const int * colors)
{
     int count = 0;
     int finished;
     for (int i = 0; i < n_vertices; ++i)
     {
         finished = 1;
         int ic = colors[i];
         for (int j = row_ptr[i]; j < row_ptr[i+1]; ++j)
         {
             if (ic == colors[col[j]]) 
             {
                 finished = 0;
                 break;
             }
         }
         if (finished == 0) continue;

         for (int j = col_ptr[i]; j < col_ptr[i+1]; ++j)
         {
             if (ic == colors[row[j]]) 
             {
                 finished = 0;
                 break;
             }
         }
         if (finished == 0) continue;
         count++;
     }
     return count;
}

inline int num_finished(const int n_vertices, 
                 const int * row_ptr, 
                 const int * col, 
                 const int * col_ptr, 
                 const int * row, 
                 const int * colors,
                 const int * reordering)
{
     int count = 0;
     int finished;
     for (int i = 0; i < n_vertices; ++i)
     {
         finished = 1;
         int ic = colors[reordering[i]];
         for (int j = row_ptr[i]; j < row_ptr[i+1]; ++j)
         {
             if (ic == colors[reordering[col[j]]]) 
             {
                 finished = 0;
                 break;
             }
         }
         if (finished == 0) continue;

         for (int j = col_ptr[i]; j < col_ptr[i+1]; ++j)
         {
             if (ic == colors[reordering[row[j]]]) 
             {
                 finished = 0;
                 break;
             }
         }
         if (finished == 0) continue;
         count++;
     }
     return count;
}

inline int compare(const int n, const int * array1, const int * array2)
{
	int count = 0;
	for(int i = 0; i < n; ++i)
        {
		if (array1[i] == array2[i])
			count++;
	}
	return count;
}

// check if all vertices are colored
__global__ void isAllColoredKernel(const int n, const int * colors, int * ret)
{
    for (int i = 0; i < n; ++i)
    {
        if (colors[i] == -1)
        {
            *ret = 0;
            return;
        }
    }
    *ret = 1;
}


__global__ void jplColorKernel(const int n, 
                                 const int c, 
                                 const int *row_ptr, 
                                 const int *col,
                                 const int *col_ptr, 
                                 const int *row,
                                 const int *randoms, 
                                 int *colors)
{   
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; 
       i < n; 
       i += blockDim.x * gridDim.x) 
  {   
    bool f=true; // true iff you have max random

    // ignore nodes colored earlier
    if ((colors[i] != -1)) continue; 

    int ir = randoms[i];

    // look at neighbors to check their random number
    for (int k = row_ptr[i]; k < row_ptr[i+1]; k++) {        
      // ignore nodes colored earlier (and yourself)
      int j = col[k];
      int jc = colors[j];
      if (((jc != -1) && (jc != c)) || (i == j)) continue; 
      int jr = randoms[j];
      if (ir <= jr) f=false;         
    }

    for (int k = col_ptr[i]; k < col_ptr[i+1]; k++) {        
      // ignore nodes colored earlier (and yourself)
      int j = row[k];
      int jc = colors[j];
      if (((jc != -1) && (jc != c)) || (i == j)) continue; 
      int jr = randoms[j];
      if (ir <= jr) f=false;         
    }

    // assign color if you have the maximum random number
    if (f) colors[i] = c;
  }
}

void jplColor(const int n_vertices, 
              const int n_edges, 
              const int *row_ptr, 
              const int *col, 
              const int * col_ptr, 
              const int * row, 
              int *colors) 
{
    int *randoms; // allocate and init random array 
    randoms = (int*)malloc(sizeof(int) * n_vertices);
    std::iota(randoms, randoms + n_vertices, 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(randoms, randoms + n_vertices, g);
    //std::copy(randoms, randoms + n_vertices, std::ostream_iterator<int>(std::cout, " "));
    //std::cout << "\n";

    thrust::fill(colors, colors + n_vertices, -1); // init colors to -1

    int * dev_randoms;
    int * dev_row_ptr;
    int * dev_col;
    int * dev_col_ptr;
    int * dev_row;
    int * dev_colors;
    cudaMalloc(&dev_randoms, sizeof(int) * n_vertices);
    cudaMalloc(&dev_row_ptr, sizeof(int) * (n_vertices + 1));
    cudaMalloc(&dev_col, sizeof(int) * n_edges);
    cudaMalloc(&dev_col_ptr, sizeof(int) * (n_vertices + 1));
    cudaMalloc(&dev_row, sizeof(int) * n_edges);
    cudaMalloc(&dev_colors, sizeof(int) * n_vertices);
    cudaMemcpy(dev_randoms, randoms, sizeof(int) * n_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_row_ptr, row_ptr, sizeof(int) * (n_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col, col, sizeof(int) * n_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col_ptr, col_ptr, sizeof(int) * (n_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_row, row, sizeof(int) * n_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_colors, colors, sizeof(int) * n_vertices, cudaMemcpyHostToDevice);
    
    int is_done;
    int * dev_is_done;
    cudaMalloc(&dev_is_done, sizeof(int));
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    int iter = 0;
    for(int c=0; c < n_vertices; c++) {
        int nt = 256;
        //int nb = min((n_vertices + nt - 1)/nt, CUDA_MAX_BLOCKS);
        int nb = (n_vertices + nt - 1)/nt; 
        jplColorKernel<<<nb, nt>>>(n_vertices, c, 
                                    dev_row_ptr, dev_col,
                                    dev_col_ptr, dev_row,
                                    dev_randoms, 
                                    dev_colors);

        //isAllColoredKernel<<<1, 1>>>(n_vertices, dev_colors, dev_is_done);
        //cudaMemcpy(&is_done, dev_is_done, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(colors, dev_colors, sizeof(int) * n_vertices, cudaMemcpyDeviceToHost);
        int num_undone = std::count(colors, colors + n_vertices, -1);
        std::cout << "Iter: " << iter << " "
                  << "num_undone = " << num_undone << " " 
                  << "time = " << time << "ms" << std::endl;
        iter++;
        //if (is_done == 1) break;
    }
    gettimeofday(&end_time, NULL);
    std::cout << "Main loop time: " << elapsed(start_time, end_time) << std::endl;
    cudaMemcpy(colors, dev_colors, sizeof(int) * n_vertices, cudaMemcpyDeviceToHost);
    cudaFree(dev_is_done);
    cudaFree(dev_randoms);
    cudaFree(dev_row_ptr);
    cudaFree(dev_col);
    cudaFree(dev_col_ptr);
    cudaFree(dev_row);
    cudaFree(dev_colors);
    free(randoms);
}

void greedyColor(const int n_vertices, 
                 const int n_edges, 
                 const int *row_ptr, 
                 const int *col, 
                 const int *col_ptr, 
                 const int *row, 
                 int *colors) 
{
  int num_colors = 0; 
  bool * color_used = (bool*)malloc(sizeof(bool) * n_vertices);
  memset(color_used, 0, sizeof(bool) * n_vertices);
  thrust::fill(colors, colors + n_vertices, -1); // init colors to -1

  struct timeval start_time, end_time;

  gettimeofday(&start_time, NULL);
    
     for (int i = 0; i < n_vertices; ++i)
  {
    memset(color_used, 0, sizeof(bool) * num_colors);
    // Mark the colors used by neighbors
    // Traverse its destinations
    for (int j = row_ptr[i]; j < row_ptr[i+1]; ++j)
    {
      int c = colors[col[j]];
      if (c != -1)
        color_used[c] = true;
    }

    // Traverse its sources 
    for (int j = col_ptr[i]; j < col_ptr[i+1]; ++j)
    {
      int c = colors[row[j]];
      if (c != -1)
        color_used[c] = true;
    }

    // select a color which can be used
    for (int j = 0; j < num_colors; ++j)
    {
      if (color_used[j] == false)
        colors[i] = j;
    }
    // add a new color if there is no suitable color
    if (colors[i] == -1)
    {
      colors[i] = num_colors++;
    }
  }

  gettimeofday(&end_time, NULL);
    std::cout << "CPU: Greeddy coloring time in function: " << elapsed(start_time, end_time) << "ms" << std::endl;
  
}

__global__ void checkAllColorsKernel(const int n, 
                             const int *row_ptr, 
                             const int *col,
                             const int *col_ptr, 
                             const int *row, 
                             const int *colors, 
                             int * finished)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; 
       i < n; i += blockDim.x * gridDim.x) 
  {
    int ic = colors[i];
    // Traverse its destinations
    for (int j = row_ptr[i]; j < row_ptr[i+1]; ++j)
    {
      int c = colors[col[j]];
      if (c == ic)
      {
          *finished = 0;
          return;
      }
    }

    // Traverse its sources 
    for (int j = col_ptr[i]; j < col_ptr[i+1]; ++j)
    {
      int c = colors[row[j]];
      if (c == ic)
      {
          *finished = 0;
          return;
      }
    }
  }
}

__global__ void color_kernel(const int n, 
                             const int new_color,
                             const int *row_ptr, 
                             const int *col,
                             const int *col_ptr, 
                             const int *row,
                             const int *pre_colors, 
                             int * ret_colors, 
                             int * continue_flag)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; 
        i < n; i += blockDim.x * gridDim.x) 
    {
        int ic = pre_colors[i];
        int update = 0;
        // Traverse its destinations
        for (int j = row_ptr[i]; j < row_ptr[i+1]; ++j)
        {
          int c = pre_colors[col[j]];
          if (c == ic && col[j] != i)
          {
              update = 1;
              break;
          }
        }

        if (update)
        {
            ret_colors[i] = new_color;
            *continue_flag = 1;
        }
        else
        {
            ret_colors[i] = pre_colors[i];
        }
    }
}

void colorOnGPU(const int n_vertices, 
                const int n_edges, 
                const int *row_ptr, 
                const int *col, 
                const int * col_ptr, 
                const int * row, 
                int *colors) 
{
    //thrust::fill(colors, colors + n_vertices, -1); // init colors to -1
    int init_num_colors = 100;
    for (int i = 0; i < n_vertices; ++i)
       colors[i] = rand() % init_num_colors;

    int * colors2 = (int*)malloc(sizeof(int) * n_vertices);

    int * dev_row_ptr;
    int * dev_col;
    int * dev_col_ptr;
    int * dev_row;
    int * dev_colors;
    int * dev_colors2;
    int * dev_continue_flag;
    cudaMalloc(&dev_row_ptr, sizeof(int) * (n_vertices + 1));
    cudaMalloc(&dev_col, sizeof(int) * n_edges);
    cudaMalloc(&dev_col_ptr, sizeof(int) * (n_vertices + 1));
    cudaMalloc(&dev_row, sizeof(int) * n_edges);
    cudaMalloc(&dev_colors, sizeof(int) * n_vertices);
    cudaMalloc(&dev_colors2, sizeof(int) * n_vertices);
    cudaMalloc(&dev_continue_flag, sizeof(int));
    cudaMemcpy(dev_row_ptr, row_ptr, sizeof(int) * (n_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col, col, sizeof(int) * n_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_col_ptr, col_ptr, sizeof(int) * (n_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_row, row, sizeof(int) * n_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_colors, colors, sizeof(int) * n_vertices, cudaMemcpyHostToDevice);
    
    int continue_flag = 1;
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    int turn_f = 0;
    int new_c = init_num_colors;
    while(continue_flag) {
        turn_f = turn_f ? 0 : 1;
        cudaMemset(dev_continue_flag, 0, sizeof(int));
        int nt = 256;
        int nb = (n_vertices + nt - 1)/nt; 
        color_kernel<<<nb, nt>>>(n_vertices, new_c,
                                    dev_row_ptr, dev_col,
                                    dev_col_ptr, dev_row,
                                    (turn_f? dev_colors: dev_colors2), 
                                    (turn_f? dev_colors2: dev_colors),
                                    dev_continue_flag);
        cudaMemcpy(&continue_flag, dev_continue_flag, sizeof(int), cudaMemcpyDeviceToHost);
        new_c++;
        if (new_c % 100 == 0) {
            std::cout << "num_colors=" << new_c << ", ";
            cudaMemcpy(colors, (turn_f?dev_colors2: dev_colors), sizeof(int) * n_vertices, cudaMemcpyDeviceToHost);
            cudaMemcpy(colors, dev_colors, sizeof(int) * n_vertices, cudaMemcpyDeviceToHost);
            cudaMemcpy(colors2, dev_colors2, sizeof(int) * n_vertices, cudaMemcpyDeviceToHost);
            std::cout << "num_finshed=" << num_finished(n_vertices, row_ptr, col, col_ptr, row, colors) << ", "
                      << "compare=" << compare(n_vertices, colors, colors2) << std::endl;
        }
    }
    gettimeofday(&end_time, NULL);
    std::cout << "Main loop time: " << elapsed(start_time, end_time) << std::endl;
    cudaMemcpy(colors, dev_colors, sizeof(int) * n_vertices, cudaMemcpyDeviceToHost);
    cudaFree(dev_continue_flag);
    cudaFree(dev_row_ptr);
    cudaFree(dev_col);
    cudaFree(dev_col_ptr);
    cudaFree(dev_row);
    cudaFree(dev_colors);
    cudaFree(dev_colors2);
    free(colors2);
}

void colorBycuSPARSE(const int n_vertices, 
                     const int n_edges, 
                     const int *row_ptr, 
                     const int *col, 
                     const int * col_ptr, 
                     const int * row, 
                     int *colors) 
{
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    checkCudaErrors(cusparseCreate(&handle));
    checkCudaErrors(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    cusparseColorInfo_t info;
    checkCudaErrors(cusparseCreateColorInfo(&info));
    
    float * val = (float*)malloc(sizeof(float) * n_edges);
    for (int i = 0; i < n_edges; ++i)
        val[i] = 1;

    int *d_row_ptr, *d_col;
    float *d_val;
    int *d_colors;
    cudaMalloc(&d_colors, sizeof(int) * n_vertices);
    cudaMalloc(&d_row_ptr, sizeof(int) * (n_vertices + 1));
    cudaMalloc(&d_col, sizeof(int) * n_edges);
    cudaMalloc(&d_val, sizeof(float) * n_edges);

    cudaMemcpy(d_row_ptr, row_ptr, sizeof(int) * (n_vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col, sizeof(int) * n_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, sizeof(float) * n_edges, cudaMemcpyHostToDevice);

    float fraction = 1;
    int ncolors = 0;

    int * reordering;
    int * d_reordering;
    reordering = (int*)malloc(sizeof(int) * n_vertices);
    cudaMalloc(&d_reordering, sizeof(int) * n_vertices);
    
    checkCudaErrors(cusparseScsrcolor(handle, n_vertices, n_edges, descr, d_val, d_row_ptr, d_col, &fraction, &ncolors, d_colors, d_reordering, info));
    cudaMemcpy(colors, d_colors, sizeof(int) * n_vertices, cudaMemcpyDeviceToHost);
    cudaMemcpy(reordering, d_reordering, sizeof(int) * n_vertices, cudaMemcpyDeviceToHost);
    std::cout << "ncolors=" << ncolors << std::endl;
    std::cout << "1 = " << reordering[1] << ", 2 = " << reordering[403098] << std::endl;
    std::cout << "color: 1 = " << colors[reordering[1]] << ", 2 = " << colors[reordering[403098]] << std::endl;
    int n_finished = num_finished(n_vertices, row_ptr, col, col_ptr, row, colors, reordering);
    std::cout << n_finished << " nodes are correct, but " << n_vertices - n_finished << " are incorrect.\n";
    cudaFree(d_row_ptr);
    cudaFree(d_col);
    cudaFree(d_val);
    cudaFree(d_colors);
}

__device__ void acquire_semaphore(volatile int *lock)
{
  while (atomicCAS((int *)lock, 0, 1) != 0);
}

__device__ void release_semaphore(volatile int *lock)
{
  *lock = 0;
  __threadfence();
}

__global__ void colorByEdgeKernel(const int n_edges, 
                                     const int * srcs, 
                                     const int * dsts, 
                                     int * locks, 
                                     int * colors,
                                     int * continue_flag)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; 
        //i < n_edges; i += blockDim.x * blockDim.y * blockDim.z * gridDim.x)
        i < n_edges; i += blockDim.x * gridDim.x)
    {
        int src = srcs[i];
        int dst = dsts[i];
        //acquire_semaphore(&(locks[src]));
        //acquire_semaphore(&(locks[dst]));
        if (colors[src] == colors[dst])
        {
            colors[dst] = colors[src] + 1;
            *continue_flag = 1;
        }
        //release_semaphore(&(locks[src]));
        //release_semaphore(&(locks[dst]));
    }
}

void colorByEdgeOnGPU(const int n_vertices,
                      const int n_edges, 
                      const int * srcs, 
                      const int * dsts, 
                      const int *row_ptr, 
                      const int *col, 
                      const int *col_ptr, 
                      const int *row,
                      int * colors)
{
    int * dev_srcs;
    int * dev_dsts;
    int * dev_colors;
    int * dev_locks;
    int * dev_continue_flag;
    int * locks = (int *)malloc(sizeof(int) * n_vertices);
    memset(colors, 0, sizeof(int) * n_vertices);
    memset(locks, 0, sizeof(int) * n_vertices);
    cudaMalloc(&dev_continue_flag, sizeof(int));
    cudaMalloc(&dev_srcs, sizeof(int) * n_edges);
    cudaMalloc(&dev_dsts, sizeof(int) * n_edges);
    cudaMalloc(&dev_colors, sizeof(int) * n_vertices);
    cudaMalloc(&dev_locks, sizeof(int) * n_vertices);
    cudaMemcpy(dev_srcs, srcs, sizeof(int) * n_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dsts, dsts, sizeof(int) * n_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_colors, colors, sizeof(int) * n_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_locks, locks, sizeof(int) * n_vertices, cudaMemcpyHostToDevice);
    
    int continue_flag = 1;

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    while(continue_flag) {
        int nt = 256;
        int nb = (n_vertices + nt - 1)/nt; 
        nb = nb > MAX_GRID_SIZE ? MAX_GRID_SIZE : nb;
        cudaMemset(dev_continue_flag, 0, sizeof(int));
        colorByEdgeKernel<<<nb, nt>>>(n_edges,
                                         dev_srcs, dev_dsts, 
                                         dev_locks, dev_colors,
                                         dev_continue_flag);

        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "getLastCudaError() CUDA error: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        cudaMemcpy(&continue_flag, dev_continue_flag, sizeof(int), cudaMemcpyDeviceToHost);
        //cudaMemcpy(colors, dev_colors, sizeof(int) * n_vertices, cudaMemcpyDeviceToHost);
        //std::cout << "num_finshed=" << num_finished(n_vertices, row_ptr, col, col_ptr, row, colors) << std::endl;
    }
    gettimeofday(&end_time, NULL);
    std::cout << "Main loop time: " << elapsed(start_time, end_time) << std::endl;
    cudaMemcpy(colors, dev_colors, sizeof(int) * n_vertices, cudaMemcpyDeviceToHost);

    cudaFree(dev_continue_flag);
    cudaFree(dev_srcs);
    cudaFree(dev_dsts);
    cudaFree(dev_colors);
    cudaFree(dev_locks);
    free(locks);
}

__global__ void mix_color_kernel(const int n_edges, 
                                 const int * srcs, 
                                 const int * dsts, 
                                 int * undone, 
                                 int * colors,
                                 int * continue_flag,
                                int * same_color_sum)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; 
        i < n_edges; i += blockDim.x * gridDim.x)
    {
        int src = srcs[i];
        int dst = dsts[i];
        if (colors[src] == colors[dst])
        {
            colors[dst] = colors[src] + 1;
            undone[dst] = 1;
            //undone[src] = 1;
            *continue_flag = 1;
            //atomicAdd(same_color_sum,1);
        }
    }
}

// color the vertices which are targets
void greedyColor2(const int n_vertices, 
                  const int *row_ptr, 
                  const int *col, 
                  const int * col_ptr, 
                  const int * row, 
                  const int start_c, 
                  const int * is_target,
                  int *colors)
{
  int end_c = start_c; 
  int num_colors = 0;
  bool * color_used = (bool*)malloc(sizeof(bool) * n_vertices);
  memset(color_used, 0, sizeof(bool) * n_vertices);
  for (int i = 0; i < n_vertices; ++i)
  {
    if (is_target[i] == 0)
        continue;
    memset(color_used + start_c, 0, sizeof(bool) * num_colors);
    // Mark the colors used by neighbors
    // Traverse its destinations
    for (int j = row_ptr[i]; j < row_ptr[i+1]; ++j)
    {
      int dst = col[j];
      if (is_target[dst] == 0) continue;

      int c = colors[dst];
      if (c >= start_c)
        color_used[c] = true;
    }

    // Traverse its sources 
    for (int j = col_ptr[i]; j < col_ptr[i+1]; ++j)
    {
      int src = row[j];
      if (is_target[src] == 0) continue;

      int c = colors[src];
      if (c >= start_c)
        color_used[c] = true;
    }

    // select a color which can be used
    for (int j = start_c; j < end_c; ++j)
    {
      if (color_used[j] == false)
        colors[i] = j;
    }
    // add a new color if there is no suitable color
    if (colors[i] < start_c)
    {
      colors[i] = end_c++;
      num_colors++;
    }
  }
}

void mixColor(const int n_vertices, 
              const int n_edges, 
              const int * srcs, 
              const int * dsts, 
              const int *row_ptr, 
              const int *col, 
              const int *col_ptr, 
              const int *row,
              const int niters,
              const float fraction,
              int * colors)
{
    struct timeval start_time, end_time;
    int * dev_srcs;
    int * dev_dsts;
    int * dev_colors;
    int * dev_undone;
    int * dev_continue_flag;
    int continue_flag = 1;
    int * undone;

    gettimeofday(&start_time, NULL);
    undone = (int *)malloc(sizeof(int) * n_vertices);
    memset(colors, 0, sizeof(int) * n_vertices);
    memset(undone, 0, sizeof(int) * n_vertices);
    cudaMalloc(&dev_srcs, sizeof(int) * n_edges);
    cudaMalloc(&dev_dsts, sizeof(int) * n_edges);
    cudaMalloc(&dev_colors, sizeof(int) * n_vertices);
    cudaMalloc(&dev_undone, sizeof(int) * n_vertices);
    cudaMalloc(&dev_continue_flag, sizeof(int));
    cudaMalloc(&same_color_sum,sizeof(int));
    gettimeofday(&end_time, NULL);
    //std::cout << "cuda malloc time: " << elapsed(start_time, end_time) << "ms" << std::endl;

    gettimeofday(&start_time, NULL);
    cudaMemcpy(dev_srcs, srcs, sizeof(int) * n_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dsts, dsts, sizeof(int) * n_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_colors, colors, sizeof(int) * n_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_undone, undone, sizeof(int) * n_vertices, cudaMemcpyHostToDevice);
    gettimeofday(&end_time, NULL);
    //std::cout << "cuda memcpy time: " << elapsed(start_time, end_time) << "ms" << std::endl;

    //gettimeofday(&start_time, NULL);
    int iter = 0;
    float kernel_time = 0;
    int same_color_sum_h;
    int lun_mix = 1;
    while(continue_flag) {
        gettimeofday(&start_time, NULL);
        int nt = 256;
        int nb = (n_vertices + nt - 1)/nt; 
        nb = nb > MAX_GRID_SIZE ? MAX_GRID_SIZE : nb;
        cudaMemset(dev_continue_flag, 0, sizeof(int));
        //cudaMemset(same_color_sum, 0, sizeof(int));
        cudaMemset(dev_undone, 0, sizeof(int) * n_vertices);
        mix_color_kernel<<<nb, nt>>>(n_edges,
                                     dev_srcs, dev_dsts, 
                                     dev_undone, dev_colors,
                                     dev_continue_flag,
                                    same_color_sum);

        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "getLastCudaError() CUDA error: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        //cudaMemcpy(&same_color_sum_h, same_color_sum, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&continue_flag, dev_continue_flag, sizeof(int), cudaMemcpyDeviceToHost);
        //printf("%d : samecolor_sum = %d\n",lun_mix,same_color_sum_h);
        gettimeofday(&end_time, NULL);
        //lun_mix++;
        float time = elapsed(start_time, end_time);
        kernel_time += time;
        cudaMemcpy(undone, dev_undone, sizeof(int) * n_vertices, cudaMemcpyDeviceToHost);
        int num_undone = std::count(undone, undone + n_vertices, 1);
        /************
        std::cout << "Iter: " << iter << " "
                  << "num_undone = " << num_undone << " " 
                  << "time = " << time << "ms" << std::endl;
        **************/
        iter++;
        if (niters > 0 && iter >= niters) break;
        if (fraction > 0 && (1 - (float)num_undone / n_vertices) >= fraction) break;
    }
    //gettimeofday(&end_time, NULL);
    //std::cout << "kernel time on gpu: " << elapsed(start_time, end_time) << "ms" << std::endl;

    gettimeofday(&start_time, NULL);
    cudaMemcpy(colors, dev_colors, sizeof(int) * n_vertices, cudaMemcpyDeviceToHost);
    if (continue_flag)
    {
        cudaMemcpy(undone, dev_undone, sizeof(int) * n_vertices, cudaMemcpyDeviceToHost);
        /***********************************************************************************
        std::cout << "After coloring on gpu: num_CC = " << getNumCC(n_vertices, row_ptr, col, col_ptr, row, undone) << ", "
                  << "num_undone = " << std::count(undone, undone + n_vertices, 1) << std::endl;
        ***********************************************************************************/
        greedyColor2(n_vertices, row_ptr, col, col_ptr, row, 1000, undone, colors) ;
    }
    gettimeofday(&end_time, NULL);
    float iter_time = 0.0;
    float trav_time = 0.0;
    trav_time = elapsed(start_time, end_time);
    iter_time = kernel_time;
    //std::cout << "Greedy color time for the rest vertices: " << elapsed(start_time, end_time) << "ms" << std::endl;
    //std::cout << "Time(without malloc & memcpy): " << kernel_time + elapsed(start_time, end_time) << "ms" << std::endl;
    std::cout << iter_time << "\t" << trav_time << "\t";



    cudaFree(dev_continue_flag);
    cudaFree(dev_srcs);
    cudaFree(dev_dsts);
    cudaFree(dev_colors);
    cudaFree(dev_undone);
    free(undone);
}
