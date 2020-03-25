#include <bits/stdc++.h>
#include <stdio.h>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"

#define NUM_TEST_FILES 1
#define GAP_SCORE -1
#define BLOCK_X_Y_DIM 32

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

// Call this kernel "qlen + tlen - 1" times, then matrix will be done.
__global__ void nw_shotgun_scoring_kernel(
  signed char * s,
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind,
  int * score_mat
) {
  // Get global and local thread index.
  int32_t g_tx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int32_t g_ty = (blockIdx.y * blockDim.y) + threadIdx.y;
  int32_t l_tx = threadIdx.x;
  int32_t l_ty = threadIdx.y;

  // Matrix dims.
  int32_t mat_w = tlen + 1;

  // Shared memory.
  __shared__ signed char s_s[16];
  __shared__ int s_score_mat[32][32];
  __shared__ char s_t[16];
  __shared__ char s_q[16];

  // Fill similarity matrix shared memory.
  if (l_ty == 0 && l_tx < 16)
    s_s[l_tx] = s[l_tx];

  // Fill score matrix shared memory.
  s_score_mat[l_ty][l_tx] = 0;
  if (g_tx > 0 && g_tx <= tlen && g_ty > 0 && g_ty <= qlen)
    s_score_mat[l_ty][l_tx] = score_mat[mat_w * g_ty + g_tx];

  // Fill target/query shared memory.
  if (g_tx > 0 && g_tx <= tlen && g_ty > 0 && g_ty <= qlen) {
    s_t[l_tx - 1] = t[g_tx - 1];
    s_q[l_ty - 1] = q[g_ty - 1];
  }

  // Ensure shared memory is filled before going on.
  __syncthreads();

  // If we are not a border thread then shotgun
  // compute the matrix, be it correct or not.
  if (g_tx > 0 && g_ty > 0) {
    int match = s_score_mat[l_ty - 1][l_tx - 1] + nw_get_sim(s_s, s_q[l_ty], s_t[l_tx]);
    int del = s_score_mat[l_ty - 1][l_tx] + mis_or_ind;
    int ins = s_score_mat[l_ty][l_tx - 1] + mis_or_ind;
    int cell = match > del ? match : del;
    cell = cell > ins ? cell : ins;
    s_score_mat[l_ty][l_tx] = cell;
  }
  __syncthreads();
  if (g_tx <= tlen && g_ty <= qlen) {
    score_mat[mat_w * g_ty + g_tx] = s_score_mat[l_ty][l_tx];
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
  signed char * s_d;
  char * t_d;
  char * q_d;
  int * score_mat_d;

  // Malloc space on GPU.
  cudaMalloc((void **) & s_d, 16 * sizeof(signed char));
  cudaMalloc((void **) & t_d, tlen * sizeof(char));
  cudaMalloc((void **) & q_d, qlen * sizeof(char));
  cudaMalloc((void **) & score_mat_d, (qlen + 1) * (tlen + 1) * sizeof(int));

  // Copy to GPU.
  cudaMemcpy(s_d, s, 16 * sizeof(signed char), cudaMemcpyHostToDevice);
  cudaMemcpy(t_d, t, tlen * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(q_d, q, qlen * sizeof(char), cudaMemcpyHostToDevice);

  // Launch compute kernel.
  dim3 GridDim(ceil((tlen + 1) / ((float) 32)), ceil((qlen + 1) / ((float) 32)), 1);
  dim3 BlockDim(32, 32, 1);

  for (uint32_t i = 0; i < qlen + tlen - 1; ++i) {
    nw_shotgun_scoring_kernel <<<GridDim, BlockDim>>>
      (s_d, t_d, q_d, tlen, qlen, mis_or_ind, score_mat_d);
    cudaDeviceSynchronize();
  }

  // TEMP! Just testing.
  int * score_mat = new int [(qlen + 1) * (tlen + 1)];
  cudaMemcpy(score_mat, score_mat_d, (qlen + 1) * (tlen + 1) * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i <= qlen; ++i) {
    for (int j = 0; j <= tlen; ++j)
      std::cout << score_mat[(qlen + 1) * i + j] << std::endl;
  }
  std::cout << "BATCH DONE" << std::endl;

  // Clean up.
  cudaFree(s_d);
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