#include <bits/stdc++.h>
#include <stdio.h>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>

#define NUM_INPUT_FILES 1

uint32_t lcs(char * X, char * Y, uint32_t m, uint32_t n) {
  uint32_t L[m + 1][n + 1];
  for (uint32_t i = 0; i <= m; i++) {
    for (uint32_t j = 0; j <= n; j++) {
      if (i == 0 || j == 0)
        L[i][j] = 0;
      else if (X[i - 1] == Y[j - 1])
        L[i][j] = L[i - 1][j - 1] + 1;
      else
        L[i][j] = (L[i - 1][j] > L[i][j - 1]) ? L[i - 1][j] : L[i][j - 1];
    }
  }
  return L[m][n];
}

int main() {
  for (uint32_t i = 0; i < NUM_INPUT_FILES; ++i) {
    std::string input_file = "datasets/" + std::to_string(i) + ".txt";
    std::ifstream input_file_stream(input_file);
    std::string input_line;
    uint32_t input_count = 0;
    uint32_t m = 0;
    uint32_t n = 0;
    char * X = NULL;
    char * Y = NULL;
    while (std::getline(input_file_stream, input_line)) {
      if (input_count == 0) {
        m = std::stoll(input_line);
        X = (char *) malloc(sizeof(char) * (m + 1));
      }
      if (input_count == 1) {
        n = std::stoll(input_line);
        Y = (char *) malloc(sizeof(char) * (n + 1));
      }
      if (input_count == 2)
        strcpy(X, input_line.c_str());
      if (input_count == 3)
        strcpy(Y, input_line.c_str());
      ++input_count;
    }
    std::cout << lcs(X, Y, m, n) << std::endl;
  }
  return 0;
}