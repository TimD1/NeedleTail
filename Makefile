CC        = g++
CFLAGS    = -std=c++11
NVCC      = nvcc
CUDA_INC  = -I/usr/local/cuda/include/
CUDA_LINK = -lcuda
CPU_SRC = nw.cpp
CPU_HDR = nw_general.hpp
GPU_SRC = nw.cu
GPU_HDR = xs.cuh xs_core.cuh cuda_error_check.cuh nw_general.hpp

gpu_nw: $(GPU_SRC) $(GPU_HDR)
	$(NVCC) $(CFLAGS) $(GPU_SRC) -o $@.o $(CUDA_LINK)

base_nw: $(CPU_SRC) $(CPU_HDR)
	$(CC) $(CFLAGS) $(CPU_SRC) -o $@.o $(CUDA_INC)

clean:
	rm -rf *.o