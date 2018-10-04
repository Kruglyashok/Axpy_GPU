#pragma once
#include <cuda_runtime_api.h>
#include <stdio.h>

__global__ void daxpy_gpu(int n, double a, double *x, int incx, double *y, int incy, double *res);
__global__ void saxpy_gpu(int n, const float a, const float *x, int incx, const float *y, int incy, float *res);