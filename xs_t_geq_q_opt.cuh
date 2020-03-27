#ifndef XS_T_GEQ_Q_OPT_CUH
#define XS_T_GEQ_Q_OPT_CUH

#include "nw_general.h"
#include "cuda_error_check.cuh"

// NOTE: XS_T_GEQ_Q => Transformation scoring
// where tlen greater than or equal to qlen.

__global__ void xs_t_geq_q_init_opt(
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
  int * score_mat
) {
  uint32_t tx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tx <= qlen + 1)
    score_mat[(tlen + 1) * tx] = tx * mis_or_ind;
  if (tx <= tlen + 1)
    score_mat[(tlen + 1) * tx + tx] = tx * mis_or_ind;
}

__global__ void xs_t_geq_q_comp_opt(
  uint32_t y_off,
  uint32_t x_off,
  uint32_t comp_w,
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
  int * score_mat
) {
  uint32_t g_tx = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint32_t l_tx = threadIdx.x;
  __shared__ int s_row_up[1025];
  // Only allow stuff in our compute width to do anything.
  if (g_tx >= comp_w)
    return;
  // Adjust our tx to correspond to
  // its real position in the matrix.
  uint32_t adj_tx = g_tx + x_off;
  uint32_t mat_w = tlen + 1;
  // Fetch the needed elements in the row above this one.
  s_row_up[l_tx] = score_mat[mat_w * (y_off - 1) + (adj_tx - 1)];
  s_row_up[l_tx + 1] = score_mat[mat_w * (y_off - 1) + adj_tx];
  s_row_up[l_tx + 1] = score_mat[mat_w * (y_off - 1) + adj_tx];
  // Get the upper left cell, post transformation.
  int match = score_mat[mat_w * (y_off - 2) + (adj_tx - 1)]
    + cuda_nw_get_sim(q[y_off - adj_tx - 1], t[adj_tx - 1]);
  // Get the upper cell, post transformation.
  int del = s_row_up[l_tx + 1] + mis_or_ind;
  // Get the left cell, post transformation.
  int ins = s_row_up[l_tx] + mis_or_ind;
  int cell = match > del ? match : del;
  cell = cell > ins ? cell : ins;
  score_mat[mat_w * y_off + adj_tx] = cell;
}

int * xs_t_geq_q_man_opt(
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
  uint32_t xf_mat_w = tlen + 1;
  uint32_t xf_mat_h = qlen + tlen + 1;

  // Malloc space on GPU.
  cuda_error_check( cudaMalloc((void **) & t_d, tlen * sizeof(char)) );
  cuda_error_check( cudaMalloc((void **) & q_d, qlen * sizeof(char)) );
  cuda_error_check( cudaMalloc((void **) & xf_mat_d, xf_mat_w * xf_mat_h * sizeof(int)) );

  // Copy to GPU.
  cuda_error_check( cudaMemcpy(t_d, t, tlen * sizeof(char), cudaMemcpyHostToDevice) );
  cuda_error_check( cudaMemcpy(q_d, q, qlen * sizeof(char), cudaMemcpyHostToDevice) );

  // Init our scoring matrix.
  uint32_t init_num_threads = tlen + 1;
  dim3 init_g_dim(ceil(init_num_threads / ((float) 1024)));
  dim3 init_b_dim(1024);
  xs_t_geq_q_init_opt <<<init_g_dim, init_b_dim>>>
    (tlen, qlen, mis_or_ind, xf_mat_d);

  // DP algorithm scoring.
  uint32_t comp_num_threads = qlen;
  dim3 comp_g_dim(ceil(comp_num_threads / ((float) 1024)));
  dim3 comp_b_dim(1024);

  uint32_t y_off = 2;
  uint32_t x_off = 1;
  uint32_t comp_w = 1;
  bool w_hit_qlen = false;
  cudaDeviceSynchronize();
  for (uint32_t wave = 0; wave < qlen + tlen - 1; ++wave) {
    // Launch kernel.
    xs_t_geq_q_comp_opt <<<comp_g_dim, comp_b_dim>>>
      (y_off, x_off, comp_w, t_d, q_d, tlen, qlen, mis_or_ind, xf_mat_d);
    // If we are going to go off the RHS of the matrix
    // reduce the width of the compute region.
    if (x_off + comp_w == tlen + 1)
      --comp_w;
    // Once we hit max band start to move
    // diagonally across the matrix.
    if (comp_w == qlen)
      w_hit_qlen = true;
    x_off += w_hit_qlen;
    comp_w += !w_hit_qlen;
    // Always move to the next row of the matrix.
    ++y_off;
    cudaDeviceSynchronize();
  }

  int * xf_mat = new int [xf_mat_w * xf_mat_h];
  cuda_error_check( cudaMemcpy(xf_mat, xf_mat_d, xf_mat_w * xf_mat_h * sizeof(int), cudaMemcpyDeviceToHost) );

  // // TEMP: UNCOMMENT FOR MATRIX PRINTING!
  // for (uint32_t i = 0; i <= qlen; ++i) {
  //   for (uint32_t j = 0; j <= tlen; ++j) {
  //     std::cout << std::setfill(' ') << std::setw(5)
  //       << xf_mat[(tlen+1) * (i+j) + j] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // Clean up.
  cudaFree(t_d);
  cudaFree(q_d);
  cudaFree(xf_mat_d);

  return xf_mat;
}

#endif