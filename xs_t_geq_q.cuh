#ifndef XS_T_GEQ_Q_CUH
#define XS_T_GEQ_Q_CUH

#include "nw_general.h"

// NOTE: XS_T_GEQ_Q => Transformation scoring
// where tlen greater than or equal to qlen.

void cuda_error_check(cudaError_t e) {
  if (e != cudaSuccess) {
    std::cerr << "CUDA FAILURE: " << cudaGetErrorString(e) << std::endl;
    exit(0);
  }
}

__global__ void xs_t_geq_q_init(
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
  int * score_mat
) {
  int32_t tx = (blockIdx.x * blockDim.x) + threadIdx.x;

}

int * xs_t_geq_q_man(
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  // Device memory pointers.
  char * t_d;
  char * q_d;
  int * xf_mat_d;

  // Transformed matrix dims.
  uint32_t xf_mat_w = (qlen + 1) < (tlen + 1) ? (qlen + 1) : (tlen + 1);
  uint32_t xf_mat_h = qlen + tlen + 1;

  // Malloc space on GPU.
  cuda_error_check( cudaMalloc((void **) & t_d, tlen * sizeof(char)) );
  cuda_error_check( cudaMalloc((void **) & q_d, qlen * sizeof(char)) );
  cuda_error_check( cudaMalloc((void **) & xf_mat_d, xf_mat_w * xf_mat_h * sizeof(int)) );

  // Copy to GPU.
  cuda_error_check( cudaMemcpy(t_d, t, tlen * sizeof(char), cudaMemcpyHostToDevice) );
  cuda_error_check( cudaMemcpy(q_d, q, qlen * sizeof(char), cudaMemcpyHostToDevice) );

  // Init our scoring matrix.
  uint32_t num_threads = (qlen + 1) > (tlen + 1) ? (qlen + 1) : (tlen + 1);
  dim3 init_g_dim(ceil(num_threads / ((float) 1024)))
  dim3 init_b_dim(1024);

  nw_scoring_xform_init_kernel <<<init_g_dim, init_b_dim>>>
    (tlen, qlen, mis_or_ind, score_mat_d);
  cudaDeviceSynchronize();


  // for (uint32_t wave_it = 0; wave_it < qlen + tlen - 1; ++wave_it) {
  //   nw_scoring_stride_kernel <<<GridDim, BlockDim>>>
  //     (wave_it, t_d, q_d, tlen, qlen, mis_or_ind, score_mat_d);
  //   cudaDeviceSynchronize();
  // }

  // // Capture computed matrix.
  // int * score_mat = new int [(qlen + 1) * (tlen + 1)];
  // cuda_error_check( cudaMemcpy(score_mat, score_mat_d, (qlen + 1) * (tlen + 1) * sizeof(int), cudaMemcpyDeviceToHost) );

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