#include <stdlib.h>
#include <iostream>

#define SIZE (100*1024*1024)

__global__ void histo_kernel(unsigned char *buffer,long size,unsigned int *histo)
{
    __shared__ unsigned int temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();

    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    int offset = blockDim.x*gridDim.x;

    while(tid < size)
    {
        atomicAdd(&(temp[buffer[tid]]),1);
        tid+=offset;
    }
    __syncthreads();

    atomicAdd(&(histo[threadIdx.x]),temp[threadIdx.x]);//这里限制了一个线程块只能是256个线程
}

int main(void)
{
    unsigned char *buffer = (unsigned char *)big_random_block(SIZE);//应该是自己写的函数
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    unsigned char *dev_buffer;
    unsigned int *dev_histo;
    cudaMalloc((void **)&dev_buffer,SIZE);
    cudaMemcpy(dev_buffer,buffer,SIZE,cudaMemcpyHostToDevice);
    cudaMalloc((void **)&dev_histo,sizeof(int)*256);
    cudaMemset(dev_histo,0,256*sizeof(int));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    int blocks = prop.multiProcessorCount;//确定线程块块数

    histo_kernel<<<blocks*2,256>>>(dev_buffer,SIZE,dev_histo);
    unsigned int histo[256];
    cudaMemcpy(histo,dev_histo,sizeof(int)*256,cudaMemcpyDeviceToHost);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,start,stop);
    printf("Time to generate: %3.1f ms\n",elapsedTime);//计算用时

    long histoCount = 0;//大致估计正确性
    for(int i=0;i<256;i++)
        histoCount+=histo[i];
    printf("Histogram Sum:%ld\n",histoCount);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_histo);
    cudaFree(dev_buffer);
    free(buffer);
    return 0;
}