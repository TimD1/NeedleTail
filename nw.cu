#include "nw_general.h"

// Call this kernel "qlen + tlen - 1" times, then matrix will be done.
__global__ void nw_shotgun_scoring_kernel(
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
  __shared__ int s_score_mat[32][32];
  __shared__ char s_t[16];
  __shared__ char s_q[16];

  // Fill score matrix shared memory.
  if (g_tx == 0 && g_ty <= qlen)
    s_score_mat[l_ty][l_tx] = g_ty * mis_or_ind;
  if (g_ty == 0 && g_tx <= tlen)
    s_score_mat[l_ty][l_tx] = g_tx * mis_or_ind;
  if (g_tx > 0 && g_tx <= tlen && g_ty > 0 && g_ty <= qlen)
    s_score_mat[l_ty][l_tx] = score_mat[mat_w * g_ty + g_tx];

  // Fill target/query shared memory.
  if (g_tx == 0 && g_ty > 0 && g_ty <= qlen)
    s_q[l_ty - 1] = q[g_ty - 1];
  if (g_ty == 0 && g_tx > 0 && g_tx <= tlen)
    s_t[l_tx - 1] = t[g_tx - 1];

  // Ensure shared memory is filled before going on.
  __syncthreads();

  // If we are not a border thread then shotgun
  // compute the matrix, be it correct or not.
  if (g_tx > 0 && g_tx <= tlen && g_ty > 0 && g_ty <= qlen) {
    int match = s_score_mat[l_ty - 1][l_tx - 1] + cuda_nw_get_sim(s_q[l_ty - 1], s_t[l_tx - 1]);
    int del = s_score_mat[l_ty - 1][l_tx] + mis_or_ind;
    int ins = s_score_mat[l_ty][l_tx - 1] + mis_or_ind;
    int cell = match > del ? match : del;
    cell = cell > ins ? cell : ins;
    s_score_mat[l_ty][l_tx] = cell;
  }

  // Coalesced writeback.
  __syncthreads();
  if (g_tx <= tlen && g_ty <= qlen)
    score_mat[mat_w * g_ty + g_tx] = s_score_mat[l_ty][l_tx];
}

int * nw_gpu_man(
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
  cudaMalloc((void **) & t_d, tlen * sizeof(char));
  cudaMalloc((void **) & q_d, qlen * sizeof(char));
  cudaMalloc((void **) & score_mat_d, (qlen + 1) * (tlen + 1) * sizeof(int));

  // Copy to GPU.
  cudaMemcpy(t_d, t, tlen * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(q_d, q, qlen * sizeof(char), cudaMemcpyHostToDevice);

  // Launch compute kernel.
  dim3 GridDim(ceil((tlen + 1) / ((float) 32)), ceil((qlen + 1) / ((float) 32)));
  dim3 BlockDim(32, 32);
  for (uint32_t i = 0; i < qlen + tlen - 1; ++i) {
    nw_shotgun_scoring_kernel <<<GridDim, BlockDim>>>
      (t_d, q_d, tlen, qlen, mis_or_ind, score_mat_d);
  }

  // Capture computed matrix.
  int * score_mat = new int [(qlen + 1) * (tlen + 1)];
  cudaMemcpy(score_mat, score_mat_d, (qlen + 1) * (tlen + 1) * sizeof(int), cudaMemcpyDeviceToHost);

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

int main() {
  // Input variables.
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
  // Write similarity matrix to constant CUDA memory.
  cudaMemcpyToSymbol(c_s, s, 16 * sizeof(signed char));

  // Prepare our time recording.
  auto start = std::chrono::high_resolution_clock::now();
  auto finish = std::chrono::high_resolution_clock::now();
  auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

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

    // Run matrix computation and time runtime.
    start = std::chrono::high_resolution_clock::now();
    int * nw_score_mat = nw_gpu_man(t, q, tlen, qlen, GAP_SCORE);
    finish = std::chrono::high_resolution_clock::now();
    runtime += std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

    // Backtrack through matrix.
    nw_backtrack(nw_score_mat, s, t, q, tlen, qlen, GAP_SCORE);

    // Clean up memory.
    delete [] nw_score_mat;
    delete [] q;
    delete [] t;
  }

  // Clean up similarity matrix memory.
  delete [] s;
  // Print out runtime and kill program.
  std::cerr << "Para Runtime: " << runtime.count() << " us" << std::endl;
  return 0;
}