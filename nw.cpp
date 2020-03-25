#include <bits/stdc++.h>
#include <stdio.h>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>

#define NUM_TEST_FILES 2
#define GAP_SCORE -1

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

int ** nw_scoring(
  signed char * s,
  char * t,
  char * q,
  uint32_t tlen,
  uint32_t qlen,
  signed char mis_or_ind
) {
  int ** mat = new int * [qlen + 1];
  for(uint32_t i = 0; i < qlen + 1; ++i)
    mat[i] = new int[tlen + 1];
  mat[0][0] = 0;
  for (uint32_t i = 1; i <= qlen; ++i)
    mat[i][0] = mis_or_ind * i;
  for (uint32_t i = 1; i <= tlen; ++i)
    mat[0][i] = mis_or_ind * i;
  for (uint32_t i = 1; i <= qlen; ++i) {
    for (uint32_t j = 1; j <= tlen; ++j) {
      int match = mat[i-1][j-1] + nw_get_sim(s, q[i-1], t[j-1]);
      int del = mat[i-1][j] + mis_or_ind;
      int ins = mat[i][j-1] + mis_or_ind;
      int cell = match > del ? match : del;
      cell = cell > ins ? cell : ins;
      mat[i][j] = cell;
    }
  }
  return mat;
}

void nw_backtrack(
  int ** mat,
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
    if (i > 0 && j > 0 && mat[i][j] == mat[i-1][j-1] + nw_get_sim(s, q[i-1], t[j-1])) {
      q_algn = q[i-1] + q_algn;
      t_algn = t[j-1] + t_algn;
      --i;
      --j;
    }
    else if (i > 0 && mat[i][j] == mat[i-1][j] + mis_or_ind) {
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
    int ** nw_score_mat = nw_scoring(s, t, q, tlen, qlen, GAP_SCORE);
    nw_backtrack(nw_score_mat, s, t, q, tlen, qlen, GAP_SCORE);
    for(int i = 0; i < qlen + 1; ++i)
      delete [] nw_score_mat[i];
    delete [] nw_score_mat;
    delete [] q;
    delete [] t;
  }
  delete [] s;
  return 0;
}