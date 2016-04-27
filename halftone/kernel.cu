
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

template<char channel>
__global__ void separate(char * input, char * output, int size) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < size) {
		output[tid] = input[3 * tid + channel];
	}
}
template __global__ void separate<0>(char *, char *, int);
template __global__ void separate<1>(char *, char *, int);
template __global__ void separate<2>(char *, char *, int);

__global__ void merge(char * in1, char * in2, char * in3, char * output, int size) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < size) {
		char * channel;
		switch(tid % 3) {
			case 0: channel = in1; break;
			case 1: channel = in2; break;
			case 2: channel = in3; break;
		}
		output[tid] = channel[tid / 3];
	}
}

__global__ void rotate(char * input, dim3 in_size, char * output, dim3 out_size, double angle) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid > out_size.x * out_size.y) return;
	int x = tid % out_size.x;
	int y = tid / out_size.y;
	int cx = x - out_size.x / 2;
	int cy = y - out_size.y / 2;
	double cs = cos(angle);
	double sn = sin(angle);
	int rx = round(cx * cs + cy * -sn);
	int ry = round(cx * sn + cy * cs);
	rx += in_size.x / 2;
	ry += in_size.y / 2;
	char pix;
	if (rx >= 0 && ry >= 0 && rx < in_size.x && ry < in_size.y) {
		pix = input[ry * in_size.x  + rx];
	} else {
		pix = 255;
	}
	output[tid] = pix;
}

__global__ void unrotate(char * input, dim3 in_size, char * output, dim3 out_size, double angle) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < out_size.x * out_size.y) return;
	int x = tid % out_size.x;
	int y = tid / out_size.x;
	int cx = x - out_size.x / 2;
	int cy = y - out_size.y / 2;
	double cs = cos(angle);
	double sn = sin(angle);
	int rx = round(cx * cs + cy * -sn);
	int ry = round(cx * sn + cy * cs);
	rx += in_size.x / 2;
	ry += in_size.y / 2;
	output[tid] = input[ry * in_size.x + rx];
}

__global__ void halftone(char * input, dim3 size) {
	__shared__ float sum;
	int global_x = blockDim.x * blockIdx.x + threadIdx.x;
	int global_y = blockDim.y * blockIdx.y + threadIdx.y;
	int global_idx = global_x + global_y * size.x;
	if (global_idx == 0) {
		sum = 0;
	}
	__syncthreads();

	if (global_x < size.x && global_y < size.y) {
		atomicAdd(&sum, (float)input[global_idx]);
	}
	__syncthreads();

	if (global_idx == 0) {
		sum /= blockDim.x * blockDim.y;
		sum = (255-sum) * blockDim.x * blockDim.y / 510;
	}
	__syncthreads();

	int mid_x = blockDim.x * blockIdx.x + blockDim.x / 2;
	int mid_y = blockDim.y * blockIdx.y + blockDim.y / 2;
	int dis_sq = (global_x - mid_x) * (global_x - mid_x) + (global_y - mid_y) * (global_y - mid_y);
	char color;
	if (dis_sq < sum) {
		color = 0;
	} else {
		color = 255;
	}
	if (global_x < size.x && global_y < size.y) {
		input[global_idx] = color;
	}
}

int main(int argc, char ** args) {
	cudaError_t cuda_status;

	if (argc != 3) {
		fprintf(stderr, "usage: \"halftone inputfilename.ext outputfilename.ext\"");
		return 1;
	}

	// read in the input image
	cv::Mat input_image = cv::imread(args[1]);
	if (input_image.data == NULL ||
			input_image.channels() != 3 ||
			input_image.depth() != CV_8U) {
		fprintf(stderr, "input file must be a 3-channel 8bit png/tiff/jpg");
		return 1;
	}
	int rows = input_image.rows;
	int cols = input_image.cols;

	// malloc buffers and memcopy input data
	char * input;
	char * channels[3];
	char * rotated[3];
	cuda_status = cudaMalloc((void**)&input, 3 * rows * cols);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto error;
	}
	for (int i = 0; i < 3; i++) {
		cuda_status = cudaMalloc((void**)&channels[i], rows * cols);
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto error;
		}
	}
	int size = rows > cols ? rows : cols;
	size = ceil(size * sqrt((double)2));
	for (int i = 0; i < 3; i++) {
		cuda_status = cudaMalloc((void**)&rotated[i], size * size);
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto error;
		}
	}
	cuda_status = cudaMemcpy(input, input_image.data, rows * cols * 3, cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto error;
	}

	// separate channels
	int maxThreads = cudaDevAttrMaxThreadsPerBlock;
	int blocks = rows * cols / maxThreads + 1;
	separate<0> <<< blocks, maxThreads >>> (input, channels[0], rows * cols);
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "separate<0> launch failed: %s\n", cudaGetErrorString(cuda_status));
		goto error;
	}
	separate<1> <<< blocks, maxThreads >>> (input, channels[1], rows * cols);
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "separate<1> launch failed: %s\n", cudaGetErrorString(cuda_status));
		goto error;
	}
	separate<2> <<< blocks, maxThreads >>> (input, channels[2], rows * cols);
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "separate<2> launch failed: %s\n", cudaGetErrorString(cuda_status));
		goto error;
	}

	// sync at end of separate kernels
	for (int i = 0; i < 3; i++) {
		cuda_status = cudaDeviceSynchronize();
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching separate!\n", cuda_status);
			goto error;
		}
	}

	// rotate channels
	double angles[3] = {0.261799, 1.309, 0.785398};
	dim3 insize(cols, rows, 1);
	dim3 outsize(size, size, 1);
	blocks = size * size / maxThreads + 1;
	for (int i = 0; i < 3; i++) {
		rotate <<< blocks, maxThreads >>> (channels[i], insize, rotated[i], outsize, angles[i]);
		cuda_status = cudaGetLastError();
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "rotate #%d launch failed: %s\n", i, cudaGetErrorString(cuda_status));
			goto error;
		}
	}

	// sync after rotate kernels
	for (int i = 0; i < 3; i++) {
		cuda_status = cudaDeviceSynchronize();
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rotate!\n", cuda_status);
			goto error;
		}
	}

	// make the halftones
	int cellsize = 12;
	int celldim = size / cellsize + 1;
	dim3 block_dims(celldim, celldim, 1);
	dim3 cell_dims(cellsize, cellsize, 1);
	dim3 boundaries(size, size, 1);
	for (int i = 0; i < 3; i++) {
		halftone <<< block_dims, cell_dims >>> (rotated[i], boundaries);
		cuda_status = cudaGetLastError();
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "halftone #%d launch failed: %s\n", i, cudaGetErrorString(cuda_status));
			goto error;
		}
	}

	// sync after halftone kernels
	for (int i = 0; i < 3; i++) {
		cuda_status = cudaDeviceSynchronize();
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching halftones!\n", cuda_status);
			goto error;
		}
	}

	// un-rotate channels
	blocks = rows * cols / maxThreads + 1;
	for (int i = 0; i < 3; i++) {
		unrotate <<< blocks, maxThreads >>> (rotated[i], outsize, channels[i], insize, -angles[i]);
		cuda_status = cudaGetLastError();
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "unrotate #%d launch failed: %s\n", i, cudaGetErrorString(cuda_status));
			goto error;
		}
	}

	// sync after unrotate kernels
	for (int i = 0; i < 3; i++) {
		cuda_status = cudaDeviceSynchronize();
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching unrotate!\n", cuda_status);
			goto error;
		}
	}

	// merge channels back into tricolor image and sync
	blocks = rows * cols * 3 / maxThreads + 1;
	merge <<< blocks, maxThreads >>> (channels[0], channels[1], channels[2], input, rows * cols * 3);
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "merge launch failed: %s\n", cudaGetErrorString(cuda_status));
		goto error;
	}
	cuda_status = cudaDeviceSynchronize();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching merge!\n", cuda_status);
		goto error;
	}

	// copy image back to host and write out to file
	cudaMemcpy(input_image.data, input, rows * cols * 3, cudaMemcpyDeviceToHost);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto error;
	}
	cv::imwrite(args[2], input_image);
	cudaFree(input);
	for (int i = 0; i < 3; i++) {
		cudaFree(channels[i]);
		cudaFree(rotated[i]);
	}
	cuda_status = cudaSuccess;

	// free memory if there was an error
error:
	cudaFree(input);
	for (int i = 0; i < 3; i++) {
		cudaFree(channels[i]);
		cudaFree(rotated[i]);
	}
	cudaDeviceReset();
	return cuda_status;
}