#include <iostream>
#include <cuda_runtime.h>
#include <bits/stdc++.h>

//data class for OHLCV
using namespace std;
class OHLCV
{
public:
        float open;
        float high;
        float low;
        float close;
        float volume;
        char date
};





// Kernel definition

__global__ void VecAdd(float* A, float* B, float* C)
(
    int i = threadIdx.x;
    C[i] = A[i] + B[i]
)

__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}