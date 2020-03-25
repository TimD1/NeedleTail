#ifndef NW_GENERAL_H
#define NW_GENERAL_H

#include <bits/stdc++.h>
#include <stdio.h>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include "cuda.h"
#include "cuda_runtime.h"

#define NUM_TEST_FILES 3
#define GAP_SCORE -1

__constant__ signed char c_s[16];

// Example similarity matrix.
//    A  G  C  T
// A  1 -1 -1 -1
// G -1  1 -1 -1
// C -1 -1  1 -1
// T -1 -1 -1  1

// Example DP Matrix
//            T
//        A  G  C  T
//     A  ..........
//  Q  G  ..........
//     C  ..........
//     T  ..........

signed char base_to_val(char B) {
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

signed char nw_get_sim(signed char * s, char Ai, char Bi) {
  return s[base_to_val(Ai) * 4 + base_to_val(Bi)];
}

__device__ signed char cuda_base_to_val(char B) {
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

__device__ signed char cuda_nw_get_sim(char Ai, char Bi) {
  return c_s[cuda_base_to_val(Ai) * 4 + cuda_base_to_val(Bi)];
}

void nw_backtrack(
  int * mat,
  signed char * s,
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  std::string t_algn = "";
  std::string q_algn = "";
  uint32_t j = tlen;
  uint32_t i = qlen;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && mat[(tlen+1) * i + j] == mat[(tlen+1) * (i-1) + (j-1)] + nw_get_sim(s, q[i-1], t[j-1])) {
      q_algn = q[i-1] + q_algn;
      t_algn = t[j-1] + t_algn;
      --i;
      --j;
    }
    else if (i > 0 && mat[(tlen+1) * i + j] == mat[(tlen+1) * (i-1) + j] + mis_or_ind) {
      q_algn = q[i-1] + q_algn;
      t_algn = '-' + t_algn;
      --i;
    }
    else {
      q_algn = '-' + q_algn;
      t_algn = t[j-1] + t_algn;
      --j;
    }
  }
  std::cout << t_algn << std::endl;
  std::cout << q_algn << std::endl;
}

#endif