#ifndef XS_CUH
#define XS_CUH

#include "nw_general.h"
#include "xs_t_geq_q.cuh"
#include "xs_t_geq_q_opt.cuh"

int * xs_man(
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  if (tlen >= qlen)
    return xs_t_geq_q_man_opt(t, q, tlen, qlen, mis_or_ind);
  else
    // return xs_t_lt_q_man(t, q, tlen, qlen, mis_or_ind);
    return NULL;
}

#endif