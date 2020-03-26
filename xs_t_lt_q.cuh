#ifndef XS_T_LT_Q_CUH
#define XS_T_LT_Q_CUH

#include "nw_general.h"

// NOTE: XS_T_LT_Q => Transformation scoring
// where tlen less than qlen.

void cuda_error_check(cudaError_t e) {
  if (e != cudaSuccess) {
    std::cerr << "CUDA FAILURE: " << cudaGetErrorString(e) << std::endl;
    exit(0);
  }
}

int * xs_t_lt_q_man(
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  return NULL;
}

#endif