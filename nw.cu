#include <bits/stdc++.h>
#include <stdio.h>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"

#define NUM_TEST_FILES 2
#define GAP_SCORE -1
#define BLOCK_X_Y_DIM 32

__constant__ signed char c_s[16];
__constant__ uint32_t    c_qlen;
__constant__ uint32_t    c_tlen;
__constant__ signed char c_mis_or_ind;
__constant__ uint32_t    c_mat_w;

__device__ signed char base_to_val(char B) {
  // Assume 'A' unless proven otherwise.
  signed char ret = 0;
  if (B == 'G')
    ret = 1;
  if (B == 'C')
    ret = 2;
  if (B == 'T')
    ret = 3;
  return ret;
}

__device__ signed char nw_get_sim(signed char * s, char Ai, char Bi) {
  return s[base_to_val(Ai) * 4 + base_to_val(Bi)];
}

__global__ void nw_scoring_kernel (
  char * t,
  char * q,
  int * score_mat
) {
  // Get thread index.
  int32_t tx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int32_t ty = (blockIdx.y * blockDim.y) + threadIdx.y;
  // Prepare score matrix.
  if (tx == 0 && ty == 0)
    score_mat[0] = 0;
  if (tx == 0 && ty > 0 && ty <= c_qlen)
    score_mat[c_mat_w * tx + ty] = ty * c_mis_or_ind;
  if (ty == 0 && tx > 0 && tx <= c_tlen)
    score_mat[c_mat_w * tx + ty] = tx * c_mis_or_ind;
  // Prepare DP loop variables.
  int match = 0;
  int del = 0;
  int ins = 0;
  int cell = 0;
  // DP compute loop.
  for (uint32_t i = 0; i < c_qlen + c_tlen - 1; ++i) {
    __syncthreads();
    if (tx > 0 && ty > 0 && tx + ty == 2 + i) {
      match = score_mat[c_mat_w * (tx - 1) + (ty - 1)] + nw_get_sim(c_s, q[ty - 1], t[tx - 1]);
      del = score_mat[c_mat_w * (tx - 1) + ty] + c_mis_or_ind;
      ins = score_mat[c_mat_w * tx + (ty - 1)] + c_mis_or_ind;
      cell = match > del ? match : del;
      cell = cell > ins ? cell : ins;
      score_mat[c_mat_w * tx + ty] = cell;
    }
  }
  __syncthreads();
  if (tx == 0 && ty == 0) {
    for (int i = 0; i < (c_qlen + 1) * (c_tlen + 1); ++i)
      printf("%d\n", score_mat[i]);
  }
}

void nw_gpu_man(
  signed char * s,
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

  // Constants.
  uint32_t mat_w = tlen + 1;

  // Malloc space on GPU.
  cudaMalloc((void **) & t_d, tlen * sizeof(char));
  cudaMalloc((void **) & q_d, qlen * sizeof(char));
  cudaMalloc((void **) & score_mat_d, (qlen+1) * (tlen+1) * sizeof(int));

  // Copy to GPU.
  cudaMemcpyToSymbol(c_s, s, 16 * sizeof(signed char));
  cudaMemcpyToSymbol(c_tlen, &tlen, sizeof(uint32_t));
  cudaMemcpyToSymbol(c_qlen, &qlen, sizeof(uint32_t));
  cudaMemcpyToSymbol(c_mat_w, &mat_w, sizeof(uint32_t));
  cudaMemcpyToSymbol(c_mis_or_ind, &mis_or_ind, sizeof(signed char));
  cudaMemcpy(t_d, t, tlen * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(q_d, q, qlen * sizeof(char), cudaMemcpyHostToDevice);

  // Launch compute kernel.
  dim3 GridDim(ceil((tlen + 1) / ((float) BLOCK_X_Y_DIM)), ceil((qlen + 1) / ((float) BLOCK_X_Y_DIM)), 1);
  dim3 BlockDim(BLOCK_X_Y_DIM, BLOCK_X_Y_DIM, 1);
  nw_scoring_kernel <<<GridDim, BlockDim>>> (t_d, q_d, score_mat_d);
  cudaDeviceSynchronize();

  std::cout << "BATCH DONE" << std::endl;

  // Clean up.
  cudaFree(t_d);
  cudaFree(q_d);
  cudaFree(score_mat_d);
}

int main() {
  std::string input_line;
  uint32_t tlen = 0;
  uint32_t qlen = 0;
  char * t = NULL;
  char * q = NULL;
  signed char * s = NULL;
  // Read in similarity matrix file.
  std::string sim_file = "datasets/similarity.txt";
  std::ifstream sim_file_stream(sim_file);
  s = new signed char[16];
  unsigned char sim_cnt = 0;
  while (std::getline(sim_file_stream, input_line)) {
    s[sim_cnt] = std::stoi(input_line);
    ++sim_cnt;
  }
  // Run through test file.
  for (uint32_t i = 0; i < NUM_TEST_FILES; ++i) {
    std::string test_file = "datasets/" + std::to_string(i) + ".txt";
    std::ifstream test_file_stream(test_file);
    uint32_t test_cnt = 0;
    while (std::getline(test_file_stream, input_line)) {
      if (test_cnt == 0) {
        tlen = std::stoll(input_line);
        t = new char [tlen + 1];
      }
      if (test_cnt == 1) {
        qlen = std::stoll(input_line);
        q = new char [qlen + 1];
      }
      if (test_cnt == 2)
        strcpy(t, input_line.c_str());
      if (test_cnt == 3)
        strcpy(q, input_line.c_str());
      ++test_cnt;
    }
    nw_gpu_man(s, t, q, tlen, qlen, GAP_SCORE);
    delete [] q;
    delete [] t;
  }
  delete [] s;
  return 0;
}