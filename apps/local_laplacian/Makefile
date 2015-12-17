include ../support/Makefile.inc

#CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp -g
CXX=g++ -O3 -march=native -Wall -fopenmp -g

local_laplacian_gen: local_laplacian_gen.cpp
	$(CXX) $(CXXFLAGS) local_laplacian_gen.cpp $(LIB_HALIDE) -o local_laplacian_gen -lpthread -ldl -lz \
	$(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

local_laplacian_ref: local_laplacian_gen
	./local_laplacian_gen 8 0

local_laplacian_auto: local_laplacian_gen
	./local_laplacian_gen 8 -1

process_ref: process.cpp local_laplacian_ref
	$(CXX) $(CXXFLAGS) process.cpp local_laplacian.o \
		-o process_ref -lpthread -ldl $(PNGFLAGS) $(OPENGL_LDFLAGS)

process_auto: process.cpp local_laplacian_auto
	$(CXX) $(CXXFLAGS) process.cpp local_laplacian.o \
		-o process_auto -lpthread -ldl $(PNGFLAGS) $(OPENGL_LDFLAGS)

out.png: process_ref
	./process_ref ../images/rgb.png 8 1 1 10 out.png

bench_ref: process_ref
	for t in 1 2 3 4 ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t \
		./process_ref ../images/rgb.png 8 1 1 10 out.png >> ref_perf.txt;\
    done

bench_auto: process_auto
	for t in 1 2 3 4 ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t \
		./process_auto ../images/rgb.png 8 1 1 10 out.png >> ref_auto.txt;\
    done

# Build rules for generating a visualization of the pipeline using HalideTraceViz
process_viz: local_laplacian_viz.o
	$(CXX) $(CXXFLAGS) process.cpp local_laplacian_viz.o -o process_viz \
		-lpthread -ldl $(PNGFLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS)

local_laplacian_viz.o: local_laplacian_gen
	HL_TRACE=3 ./local_laplacian_gen 6 0
	mv local_laplacian.o local_laplacian_viz.o

local_laplacian.mp4: process_viz
	bash viz.sh

clean:
	rm -f out.png process_ref process_auto local_laplacian.o process_viz \
		local_laplacian_viz.o local_laplacian_gen \
		local_laplacian.mp4 ref_perf.txt ref_auto.txt local_laplacian.h
