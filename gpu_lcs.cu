#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_error.h"

#define BLOCK_SIZE      1024
#define NUM_INPUT_FILES 1

__global__ void lcs_compute_kernel(char * X, char * Y, uint32_t m, uint32_t n) {

}


uint32_t lcs_kernel_manager(char * X, char * Y, uint32_t m, uint32_t n) {

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
    std::cout << lcs_kernel_manager(X, Y, m, n) << std::endl;
  }
  return 0;
}