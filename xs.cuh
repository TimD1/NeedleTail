#ifndef XS_CUH
#define XS_CUH

#include "nw_general.h"
#include "xs_t_geq_q.cuh"
#include "cuda_error_check.cuh"

int * xs_man(
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  if (tlen >= qlen) {
    uint64_t num_GPU_mem_bytes = 3 * (tlen + 1) * sizeof(int);
    num_GPU_mem_bytes += (tlen + 1) * (qlen + 1) * sizeof(int);
    num_GPU_mem_bytes += tlen * sizeof(char);
    num_GPU_mem_bytes += qlen * sizeof(char);
    // Malloc memory for our program.
    void * GPU_mem = NULL;
    cuda_error_check( cudaMalloc((void **) & GPU_mem, num_GPU_mem_bytes) );
    // Create a stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int * mat = xs_t_geq_q_man(t, q, tlen, qlen, mis_or_ind, GPU_mem, &stream);
    cudaStreamSynchronize(stream);
    cudaFree(GPU_mem);

    // // TEMP: UNCOMMENT FOR MATRIX PRINTING!
    // for (int i = 0; i <= qlen; ++i) {
    //   for (int j = 0; j <= tlen; ++j)
    //     std::cout << std::setfill(' ') << std::setw(5)
    //       << mat[(tlen+1) * i + j] << " ";
    //   std::cout << std::endl;
    // }

    return mat;
  }
  else
    return NULL;
}

#endif