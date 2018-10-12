#pragma once
#include "time.h"
#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <iostream>
#include <omp.h>
#include "gpu_axpy.cuh"
// BLAS incy, incx == 1

using namespace std;

template<typename T> bool checkResult(T* vec1, T* vec2, int n) {
	for (int i = 0; i < n; ++i) 
		if (vec1[i] - vec2[i] > 0.5) return false;
	return true;
}

void saxpy_omp(int n, float a, float *x, int incx, float *y, int incy, float *res) {
	#pragma omp parallel 
	{
	#pragma omp for schedule(guided)
		for (int i = 0; i < n; ++i) {
			if ( (i*incx < n) && (i*incy < n) )	res[i*incy] = y[i*incy] + a*x[i*incx];
		}
	}
}

void daxpy_omp(int n, double a, double *x, int incx, double *y, int incy, double *res) {
	#pragma omp parallel 
	{
	#pragma omp for schedule(guided)
		for (int i = 0; i < n; ++i) {
			if ( (i*incx < n) && (i*incy < n) )	res[i*incy] = y[i*incy] + a*x[i*incx];
		}
	}
}


void saxpy(int n, float a, float *x, int incx, float *y, int incy, float *res) {
	for (int i = 0; i*incy < n; ++i) {
		if (i*incx < n)	res[i*incy] = y[i*incy] + a*x[i*incx];
	}
}

void daxpy(int n, double a, double *x, int incx, double *y, int incy, double *res) {
	for (int i = 0; i*incy < n; ++i) {
		if (i*incx < n)	res[i*incy] = y[i*incy] + a*x[i*incx];
	}
}

template<typename T> void showVec(int n, T* vec) {
	for (int i = 0; i < n; ++i) {
		printf("%f\t", vec[i]);
	}
	printf("\n");
}

int main(int argc, char **argv) {
	int N;
	//float
	float a, *x, *x_gpu, *y, *y_gpu, *res, *res_omp, *res_gpu, *res_gpu_host, time_cpu_f, time_gpu_f, time_omp_f;
	double  *x_d, *x_gpu_d, *y_d, *y_gpu_d, *res_d, *res_omp_d, *res_gpu_d, *res_gpu_host_d, time_cpu_d, time_gpu_d, time_omp_d;

	int incx, incy, blockSize;

	N = atoi(argv[1]); // number of elements in the vector

	x = (float*)malloc(N * sizeof(float));
	y = (float*)malloc(N * sizeof(float));
	res = (float*)malloc(N * sizeof(float));
	res_gpu_host = (float*)malloc(N * sizeof(float));
	res_omp = (float*)malloc(N * sizeof(float));

	cudaMalloc((void**)&x_gpu, N * sizeof(float));
	cudaMalloc((void**)&y_gpu, N * sizeof(float));
	cudaMalloc((void**)&res_gpu, N * sizeof(float));

	x_d = (double*)malloc(N * sizeof(double));
	y_d = (double*)malloc(N * sizeof(double));
	res_d = (double*)malloc(N * sizeof(double));
	res_gpu_host_d = (double*)malloc(N * sizeof(double));
	res_omp_d = (double*)malloc(N * sizeof(double));

	cudaMalloc((void**)&x_gpu_d, N * sizeof(double));
	cudaMalloc((void**)&y_gpu_d, N * sizeof(double));
	cudaMalloc((void**)&res_gpu_d, N * sizeof(double));


	incx = atoi(argv[2]);  //increment for x
	incy = atoi(argv[3]); //increment for y
	a = atof(argv[4]); // coefficient for sum of vecs
	blockSize = atoi(argv[5]); //size of cuda blocks

	//struct cudaDeviceProp properties;
	//cudaGetDeviceProperties(&properties, 0);
	//cout << "using " << properties.multiProcessorCount << " multiprocessors" << endl;
	//cout << "max threads per processor: " << properties.maxThreadsPerMultiProcessor << endl;
	//printf("%.2f\n", a);
	for (int i = 0; i < N; ++i) {
		x[i] = (rand() % 100);
		y[i] = (rand() % 100);

		res[i] = (float)0;
		res_gpu_host[i] = (float)0;
	}
	for (int i = 0; i < N; ++i) {
		x_d[i] = (double)x[i];
		y_d[i] = (double)y[i];
	}
	clock_t start_time = clock();
	saxpy(N, a, x, incx, y, incy, res);
	clock_t end_time = clock();
	time_cpu_f = (float)(end_time - start_time) / CLOCKS_PER_SEC;
	
	start_time = clock();
	daxpy(N, a, x_d, incx, y_d, incy, res_d);
	end_time = clock();
	time_cpu_d = (double)(end_time - start_time) / CLOCKS_PER_SEC;

	start_time = clock();
	daxpy_omp(N, a, x_d, incx, y_d, incy, res_omp_d);
	end_time = clock();
	time_omp_d = (double)(end_time - start_time) / CLOCKS_PER_SEC;

	start_time = clock();
	cudaMemcpy(x_gpu_d, x_d, N*(sizeof(double)), cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu_d, y_d, N*(sizeof(double)), cudaMemcpyHostToDevice);
	daxpy_gpu<<<(blockSize + N) / blockSize, blockSize >> > (N, a, x_gpu_d, incx, y_gpu_d, incy, res_gpu_d);
	cudaMemcpy(res_gpu_host_d, res_gpu_d, N * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	end_time = clock();
	time_gpu_d = (float)(end_time - start_time) / CLOCKS_PER_SEC;

	start_time = clock();
	cudaMemcpy(x_gpu, x, N*(sizeof(float)), cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, N*(sizeof(float)), cudaMemcpyHostToDevice);
	saxpy_gpu<<<(blockSize + N) / blockSize, blockSize>>> (N, a, x_gpu, incx, y_gpu, incy, res_gpu);
	cudaMemcpy(res_gpu_host, res_gpu, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	end_time = clock();

	//printf(cudaGetErrorString(cudaGetLastError()));
	printf("\n");
	time_gpu_f = (float)(end_time - start_time) / CLOCKS_PER_SEC;

	start_time = clock();
	saxpy_omp(N, a, x, incx, y, incy, res_omp);
	end_time = clock();
	time_omp_f = (float)(end_time - start_time) / CLOCKS_PER_SEC;

	/*printf("x:\n");
	showVec(N, x);
	printf("y:\n");
	showVec(N, y);
	printf("res:\n");
	showVec(N, res);
	printf("res_gpu_host:\n");
	showVec(N, res_gpu_host);*/
	printf("time_f: %6.4g\n", time_cpu_f);
	printf("time_gpu_f: %10.4g\n", time_gpu_f);
	printf("time_omp_f: %10.4g\n", time_omp_f);
	printf(checkResult(res_omp, res, N) ? "omp_eq\n" : "omp not eq\n");
	printf(checkResult(res, res_gpu_host, N) ? "Equal\n": "Not Equal\n");
	printf("time_d: %6.4g\n", time_cpu_d);
	printf("time_gpu_d: %10.4g\n", time_gpu_d);
	printf("time_omp_d: %10.4g\n", time_omp_d);
	printf(checkResult(res_omp_d, res_d, N) ? "omp_eq\n" : "omp not eq\n");
	printf(checkResult(res_d, res_gpu_host_d, N) ? "Equal\n" : "Not Equal\n");
	cudaFree(x_gpu_d); cudaFree(y_gpu_d); cudaFree(res_gpu_d);
	free(x_d); free(y_d); free(res_d);
	cudaFree(x_gpu_d); cudaFree(y_gpu_d); cudaFree(res_gpu_d);
	free(x_d); free(y_d); free(res_d);
	return 0;
}