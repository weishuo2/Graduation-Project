#include <stdlib.h>
#include <iostream>

#define imin(a,b) (a<b?a:b)

const int N = 33*1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32,(N+threadsPerBlock-1)/threadsPerBlock);

__global__ void dot(float *a,float *b,float *c)
{
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int cacheIndex = threadIdx.x;

    float tmp = 0;
    while(tid < N)
    {
        tmp += a[tid]*b[tid];//中间变量的加法尽量在中间变量上做
        tid += blockDim.x*gridDim.x;
    }
    cache[cacheIndex] =tmp;
    __syncthreads();
    int i=blockDim.x/2;
    while(i)
    {
        if(cacheIndex < i)
        {
            cache[cacheIndex] += cache[cacheIndex+i];
        }
        __syncthreads();
        i/=2;
    }
    if(cacheIndex == 0)
    {//记录每一个block的结果到返回数组中
        c[blockIdx.x] = cache[0];
    }
}

int main(void)
{
    float *a,*b,c,*partial_c;
    float *dev_a,*dev_b,*dev_partial_c;

    a=(float*)malloc(N*sizeof(float));
    b=(float*)malloc(N*sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid*sizeof(float));

    cudaMalloc((void**)&dev_a,N*sizeof(float));
    cudaMalloc((void**)&dev_b,N*sizeof(float));
    cudaMalloc((void**)&dev_partial_c,blocksPerGrid*sizeof(float));

    for(int i=0;i<N;i++)
    {
        a[i] = i;
        b[i] = i*2;
    }

    cudaMemcpy(dev_a,a,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b,b,N*sizeof(float),cudaMemcpyHostToDevice);
    dot<<<blocksPerGrid,threadsPerBlock>>>(dev_a,dev_b,dev_partial_c);
    cudaMemcpy(partial_c,dev_partial_c,N*sizeof(float),cudaMemcpyDeviceToHost);

    c=0;
    for(int i=0;i<blocksPerGrid;i++)
    {
        c += partial_c[i];
    }
    #define sum_squares(x) (x*(x+1)*(2*x+1)/6)
    printf("Does GPU value %.6g = %.6g?\n",c,2*sum_squares((float)(N-1)));
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);
    free(a);
    free(b);
    free(partial_c);
}

