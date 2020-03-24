CC=g++
CFLAGS=-c

base_nw: base_nw_o
	g++ -o base_nw nw.o

base_nw_o: nw.cpp
	$(CC) $(CFLAGS) nw.cpp

clean:
	rm -rf *.o base_nw