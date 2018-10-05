#include "gpu_axpy.cuh"

__global__ void saxpy_gpu(int n, const float a, const float *x, int incx, const float *y, int incy, float *res) {
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i == 0) printf("a:%g\n", a);
	if (i*incy < n && i*incx < n)	res[i*incy] = a*x[i*incx] + y[i*incy];
}

__global__ void daxpy_gpu(int n, double a, double *x, int incx, double *y, int incy, double *res) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i*incy < n && i*incx < n)	res[i*incy] = a*x[i*incx] + y[i*incy];
}