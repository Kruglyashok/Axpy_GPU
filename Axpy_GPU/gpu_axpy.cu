#include "gpu_axpy.cuh"

__global__ void saxpy_gpu(int n, const float a, const float *x, int incx, const float *y, int incy, float *res) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)	res[i] = x[i] + y[i];
}

__global__ void daxpy_gpu(int n, double a, double *x, int incx, double *y, int incy, double *res) {}