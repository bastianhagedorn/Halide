include ../support/Makefile.inc

all: test

CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp
#CXX=g++ -O3 -march=native -Wall -openmp

halide_blur: halide_blur.cpp
	$(CXX) $(CXXFLAGS) halide_blur.cpp $(LIB_HALIDE) -o halide_blur -ldl -lpthread -lz $(LDFLAGS) -ltinfo

halide_blur.o: halide_blur
	./halide_blur

# g++ on OS X might actually be system clang without openmp
CXX_VERSION=$(shell $(CXX) --version)
ifeq (,$(findstring clang,$(CXX_VERSION)))
OPENMP_FLAGS=-fopenmp
else
OPENMP_FLAGS=
endif

# -O2 is faster than -O3 for this app (O3 unrolls too much)
test: test.cpp halide_blur.o halide_gen_blur.cpp
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) test.cpp halide_blur.o halide_gen_blur.cpp -o test -lpthread -ldl $(PNGFLAGS)

bench: test
	OMP_NUM_THREADS=1 HL_NUM_THREADS=1 ./test
	OMP_NUM_THREADS=2 HL_NUM_THREADS=2 ./test
	OMP_NUM_THREADS=4 HL_NUM_THREADS=4 ./test
	OMP_NUM_THREADS=8 HL_NUM_THREADS=8 ./test
	#HL_NUM_THREADS=1 numactl --physcpubind=0 ./test
	#HL_NUM_THREADS=4 numactl --physcpubind=0-3 ./test
	#HL_NUM_THREADS=8 numactl --physcpubind=0-7 ./test
	#HL_NUM_THREADS=16 numactl --physcpubind=0-15 ./test
	#HL_NUM_THREADS=32 numactl --physcpubind=0-31 ./test
	#HL_NUM_THREADS=64 numactl --physcpubind=0-63 ./test

clean:
	rm -f test halide_blur.o halide_blur
