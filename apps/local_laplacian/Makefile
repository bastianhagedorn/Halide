include ../support/Makefile.inc

CXX=icpc -march=native -ansi-alias -ipo -O3 -Wall -g -openmp
#CXX=g++ -march=native -O3 -Wall -fopenmp

all: process

local_laplacian_gen: local_laplacian_gen.cpp
	$(CXX) $(CXXFLAGS) local_laplacian_gen.cpp -g $(LIB_HALIDE) -o local_laplacian_gen -lpthread -ldl -lz \
	$(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

local_laplacian.o: local_laplacian_gen
	./local_laplacian_gen

process: process.cpp local_laplacian.o local_laplacian_c.cpp
	$(CXX) $(CXXFLAGS) process.cpp local_laplacian_c.cpp local_laplacian.o -o process -lpthread -ldl $(PNGFLAGS) $(OPENGL_LDFLAGS)

out.png: process
	./process ../images/rgb.png 8 1 1 10 out.png

bench: process
	HL_NUM_THREADS=1  ./process ../images/rgb.png 8 1 1 10 out.png
	HL_NUM_THREADS=2  ./process ../images/rgb.png 8 1 1 10 out.png
	HL_NUM_THREADS=4  ./process ../images/rgb.png 8 1 1 10 out.png
	HL_NUM_THREADS=8  ./process ../images/rgb.png 8 1 1 10 out.png
	#HL_NUM_THREADS=1 numactl --physcpubind=0 ./process ../images/rgb.png 8 1 1 10 out.png
	#HL_NUM_THREADS=4 numactl --physcpubind=0-3 ./process ../images/rgb.png 8 1 1 10 out.png
	#HL_NUM_THREADS=8 numactl --physcpubind=0-7 ./process ../images/rgb.png 8 1 1 10 out.png
	#HL_NUM_THREADS=16 numactl --physcpubind=0-15 ./process ../images/rgb.png 8 1 1 10 out.png
	#HL_NUM_THREADS=32 numactl --physcpubind=0-31 ./process ../images/rgb.png 8 1 1 10 out.png
	#HL_NUM_THREADS=64 numactl --physcpubind=0-63 ./process ../images/rgb.png 8 1 1 10 out.png

# Build rules for generating a visualization of the pipeline using HalideTraceViz
process_viz: local_laplacian_viz.o
	$(CXX) $(CXXFLAGS) process.cpp local_laplacian_viz.o -o process_viz -lpthread -ldl $(PNGFLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS)

local_laplacian_viz.o: local_laplacian_gen
	HL_TRACE=3 ./local_laplacian_gen 6
	mv local_laplacian.o local_laplacian_viz.o

local_laplacian.mp4: process_viz
	bash viz.sh

clean:
	rm -f process local_laplacian.o process_viz local_laplacian_viz.o local_laplacian_gen local_laplacian.mp4
