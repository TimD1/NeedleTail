CC=g++
CFLAGS=-c

base_lcs: base_lcs_o
	g++ -o base_lcs base_lcs.o

base_lcs_o: base_lcs.cpp
	$(CC) $(CFLAGS) base_lcs.cpp

clean:
	rm -rf *.o base_lcs