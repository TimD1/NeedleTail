#ifndef NW_SCORING_STRIDE_CUH
#define NW_SCORING_STRIDE_CUH

#include "nw_general.h"

void cuda_error_check(cudaError_t e) {
  if (e != cudaSuccess) {
    std::cerr << "CUDA FAILURE: " << cudaGetErrorString(e) << std::endl;
    exit(0);
  }
}

__global__ void nw_scoring_stride_init(
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
  int * score_mat
) {
  int32_t tx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int32_t ty = (blockIdx.y * blockDim.y) + threadIdx.y;
  int32_t mat_w = tlen + 1;
  if (ty == 0 && tx <= tlen)
    score_mat[mat_w * ty + tx] = tx * mis_or_ind;
  if (tx == 0 && ty <= qlen)
    score_mat[mat_w * ty + tx] = ty * mis_or_ind;
}

__global__ void nw_scoring_stride_kernel(
  uint32_t wave_it,
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
  int * score_mat
) {
  int32_t tx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int32_t ty = (blockIdx.y * blockDim.y) + threadIdx.y;
  int32_t mat_w = tlen + 1;
  if (tx > 0 && ty > 0 && tx + ty == 2 + wave_it) {
    int match = score_mat[mat_w * (ty - 1) + (tx - 1)] + cuda_nw_get_sim(q[ty - 1], t[tx - 1]);
    int del = score_mat[mat_w * (ty - 1) + tx] + mis_or_ind;
    int ins = score_mat[mat_w * ty + (tx - 1)] + mis_or_ind;
    int cell = match > del ? match : del;
    cell = cell > ins ? cell : ins;
    score_mat[mat_w * ty + tx] = cell;
  }
}

int * nw_scoring_stride_man(
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  // Device memory pointers.
  char * t_d;
  char * q_d;
  int * score_mat_d;

  // Malloc space on GPU.
  cuda_error_check( cudaMalloc((void **) & t_d, tlen * sizeof(char)) );
  cuda_error_check( cudaMalloc((void **) & q_d, qlen * sizeof(char)) );
  cuda_error_check( cudaMalloc((void **) & score_mat_d, (qlen + 1) * (tlen + 1) * sizeof(int)) );

  // Copy to GPU.
  cuda_error_check( cudaMemcpy(t_d, t, tlen * sizeof(char), cudaMemcpyHostToDevice) );
  cuda_error_check( cudaMemcpy(q_d, q, qlen * sizeof(char), cudaMemcpyHostToDevice) );

  // Launch compute kernel.
  dim3 GridDim(ceil((tlen + 1) / ((float) 32)), ceil((qlen + 1) / ((float) 32)));
  dim3 BlockDim(32, 32);
  nw_scoring_stride_init <<<GridDim, BlockDim>>>
    (tlen, qlen, mis_or_ind, score_mat_d);
  cudaDeviceSynchronize();

  for (uint32_t wave_it = 0; wave_it < qlen + tlen - 1; ++wave_it) {
    nw_scoring_stride_kernel <<<GridDim, BlockDim>>>
      (wave_it, t_d, q_d, tlen, qlen, mis_or_ind, score_mat_d);
    cudaDeviceSynchronize();
  }

  // Capture computed matrix.
  int * score_mat = new int [(qlen + 1) * (tlen + 1)];
  cuda_error_check( cudaMemcpy(score_mat, score_mat_d, (qlen + 1) * (tlen + 1) * sizeof(int), cudaMemcpyDeviceToHost) );

  // // TEMP: UNCOMMENT FOR MATRIX PRINTING!
  // for (int i = 0; i <= qlen; ++i) {
  //   for (int j = 0; j <= tlen; ++j)
  //     std::cout << std::setfill(' ') << std::setw(5)
  //       << score_mat[(tlen + 1) * i + j] << " ";
  //   std::cout << std::endl;
  // }

  // Clean up.
  cudaFree(t_d);
  cudaFree(q_d);
  cudaFree(score_mat_d);

  return score_mat;
}

#endif