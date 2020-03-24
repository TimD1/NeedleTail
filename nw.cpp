#include <bits/stdc++.h>
#include <stdio.h>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>

#define NUM_TEST_FILES 1

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
  if (B == 'A')
    return 0;
  if (B == 'G')
    return 1;
  if (B == 'C')
    return 2;
  if (B == 'T')
    return 3;
}

signed char nw_get_sim(signed char * s, char Ai, char Bi) {
  return s[base_to_val(Ai) * 4 + base_to_val(Bi)];
}

int * nw_scoring(signed char * s, char * t, char * q, uint32_t tlen, uint32_t qlen, signed char mis_or_ind) {
  int * mat = (int *) malloc((tlen + 1) * (qlen + 1) * sizeof(int));
  mat[0] = 0;
  for (uint32_t i = 1; i <= qlen; ++i)
    mat[i * (tlen + 1)] = mis_or_ind * i;
  for (uint32_t i = 1; i <= tlen; ++i)
    mat[i] = mis_or_ind * i;
  for (uint32_t i = 1; i <= qlen; ++i) {
    for (uint32_t j = 1; j <= tlen; ++j) {
      int m = mat[(i - 1) * (tlen + 1) + (j - 1)] + nw_get_sim(s, q[i - 1], t[j - 1]);
      int d = mat[(i - 1) * (tlen + 1) + j] + mis_or_ind;
      int i = mat[i * (tlen + 1) + (j - 1)] + mis_or_ind;
      int f = m > d ? m : d;
      f = f > i ? f : i;
      mat[i * (tlen + 1) + j] = f;
    }
  }
  return mat;
}

void nw_backtrack() {

// AlignmentA ← ""
// AlignmentB ← ""
// i ← length(A)
// j ← length(B)
// while (i > 0 or j > 0) {
//   if (i > 0 and j > 0 and F(i,j) == F(i-1,j-1) + S(Ai, Bj)) {
//     AlignmentA ← Ai + AlignmentA
//     AlignmentB ← Bj + AlignmentB
//     i ← i - 1
//     j ← j - 1
//   }
//   else if (i > 0 and F(i,j) == F(i-1,j) + d) {
//     AlignmentA ← Ai + AlignmentA
//     AlignmentB ← "-" + AlignmentB
//     i ← i - 1
//   }
//   else {
//     AlignmentA ← "-" + AlignmentA
//     AlignmentB ← Bj + AlignmentB
//     j ← j - 1
//   }
// }

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
  s = (signed char *) malloc(16 * sizeof(signed char));
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
        t = (char *) malloc(sizeof(char) * (tlen + 1));
      }
      if (test_cnt == 1) {
        qlen = std::stoll(input_line);
        q = (char *) malloc(sizeof(char) * (qlen + 1));
      }
      if (test_cnt == 2)
        strcpy(t, input_line.c_str());
      if (test_cnt == 3)
        strcpy(q, input_line.c_str());
      ++test_cnt;
    }
    int * nw_score_mat = nw_scoring(s, t, q, tlen, qlen, -1);
  }
  return 0;
}