gpu_nw: nw.cu
	nvcc nw.cu -o gpu_nw

base_nw: base_nw_o
	g++ -o base_nw nw.o

base_nw_o: nw.cpp
	g++ -c nw.cpp

clean:
	rm -rf *.o base_nw gpu_nw