CC        = g++
NVCC      = nvcc
BASE_MAIN = nw.cpp
BASE_DEPS = nw.cpp nw_general.h
CUDA_MAIN = nw.cu
CUDA_DEPS = nw.cu nw_general.h nw_scoring_stride.cuh

gpu_nw: $(CUDA_DEPS)
	$(NVCC) $(CUDA_MAIN) -o $@.o

base_nw: $(BASE_DEPS)
	$(CC) $(BASE_MAIN) -o $@.o

gpu_nw_debug: $(CUDA_DEPS)
	$(NVCC) -G -g $(CUDA_MAIN) -o $@.o

clean:
	rm -rf *.o