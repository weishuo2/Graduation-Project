#include <stdlib.h>
#include <iostream>

using namespace std;

__global__ void Plus(float A[], float B[], float C[], int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    float*A, *Ad, *B, *Bd, *C, *Cd;
    int n = 1024 * 1024;
    int size = n * sizeof(float);

    // CPU端分配内存
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // 初始化数组
    for(int i=0;i<n;i++)
    {
        A[i] = 90.0;
        B[i] = 10.0;
    }

    // GPU端分配内存
    cudaMalloc((void**)&Ad, size);
    cudaMalloc((void**)&Bd, size);
    cudaMalloc((void**)&Cd, size);

    // CPU的数据拷贝到GPU端
    cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

    // 定义kernel执行配置，（1024*1024/512）个block，每个block里面有512个线程
    dim3 dimBlock(512);
    dim3 dimGrid(n/512);

    // 执行kernel
    Plus<<<dimGrid, dimBlock>>>(Ad, Bd, Cd, n);

    // 将在GPU端计算好的结果拷贝回CPU端
    cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);

    // 校验误差
    float max_error = 0.0;
    for(int i=0;i<n;i++)
    {
        max_error += fabs(100.0 - C[i]);
    }

    cout << "max error is " << max_error << endl;

    // 释放CPU端、GPU端的内存
    free(A);
    free(B);
    free(C);
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    return 0;
}