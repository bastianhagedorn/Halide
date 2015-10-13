include ../support/Makefile.inc

all: filter

bilateral_grid: bilateral_grid.cpp
	$(CXX) $(CXXFLAGS) bilateral_grid.cpp -g $(LIB_HALIDE) -o bilateral_grid -lpthread -ldl -lz $(LDFLAGS) \
	$(LLVM_SHARED_LIBS) -ltinfo

bilateral_grid.o: bilateral_grid
	./bilateral_grid 8

filter: bilateral_grid.o filter.cpp
	$(CXX) $(CXXFLAGS) -O3 -ffast-math -Wall -Werror filter.cpp bilateral_grid.o -lpthread -ldl -o filter  $(PNGFLAGS)

bilateral_grid.mp4: bilateral_grid.cpp viz.sh
	bash viz.sh

out.png: filter
	./filter ../images/gray.png out.png 0.1 10

bench: filter
	HL_NUM_THREADS=1 numactl --physcpubind=0 ./filter ../images/gray.png out.png 0.1 10
	HL_NUM_THREADS=4 numactl --physcpubind=0-3 ./filter ../images/gray.png out.png 0.1 10
	HL_NUM_THREADS=8 numactl --physcpubind=0-7 ./filter ../images/gray.png out.png 0.1 10
	HL_NUM_THREADS=16 numactl --physcpubind=0-15 ./filter ../images/gray.png out.png 0.1 10
	HL_NUM_THREADS=32 numactl --physcpubind=0-31 ./filter ../images/gray.png out.png 0.1 10
	HL_NUM_THREADS=64 numactl --physcpubind=0-63 ./filter ../images/gray.png out.png 0.1 10

clean:
	rm -f bilateral_grid bilateral_grid.mp4 bilateral_grid.o bilateral_grid.h bilateral filter
