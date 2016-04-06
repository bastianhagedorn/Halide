TESTS=process_ref process_auto process_naive process_sweep process_rand
include ../support/Makefile.inc

auto: clean_auto process_auto

local_laplacian_gen: local_laplacian_gen.cpp
	$(CXX) $(CXXFLAGS) local_laplacian_gen.cpp $(LIB_HALIDE) -o local_laplacian_gen -lpthread -ldl -lz \
	$(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

local_laplacian_ref.o: local_laplacian_gen
	./local_laplacian_gen 8 0

local_laplacian_auto.o: local_laplacian_gen
	./local_laplacian_gen 8 -1

local_laplacian_naive.o: local_laplacian_gen
	HL_AUTO_NAIVE=1 ./local_laplacian_gen 8 -1

local_laplacian_naive_gpu.o: local_laplacian_gen
	HL_AUTO_NAIVE=1 HL_AUTO_GPU=1 ./local_laplacian_gen 8 -2

local_laplacian_auto_gpu.o: local_laplacian_gen
	HL_AUTO_GPU=1 ./local_laplacian_gen 8 -2

local_laplacian_rand.o: local_laplacian_gen
	HL_AUTO_RAND=1 HL_AUTO_PARALLELISM=8 HL_AUTO_VEC_LEN=16 HL_AUTO_BALANCE=15 HL_AUTO_FAST_MEM_SIZE=1048576 ./local_laplacian_gen 8 -1
	#HL_AUTO_RAND=1 HL_AUTO_PARALLELISM=8 HL_AUTO_VEC_LEN=8 HL_AUTO_BALANCE=10 HL_AUTO_FAST_MEM_SIZE=131072 ./local_laplacian_gen 8 -1

local_laplacian_sweep.o: local_laplacian_gen
	HL_AUTO_SWEEP=1 HL_AUTO_PARALLELISM=12 HL_AUTO_VEC_LEN=16 HL_AUTO_BALANCE=20 HL_AUTO_FAST_MEM_SIZE=131072 ./local_laplacian_gen 8 -1

process_%: process.cpp local_laplacian_%.o
	$(CXX) $(CXXFLAGS) $^ \
		-o $@ -lpthread -ldl $(PNGFLAGS) $(OPENGL_LDFLAGS)

out.png: process_ref
	./process_ref ../images/rgb.png 8 1 1 10 out.png

# Build rules for generating a visualization of the pipeline using HalideTraceViz
process_viz: local_laplacian_viz.o
	$(CXX) $(CXXFLAGS) process.cpp local_laplacian_viz.o -o process_viz \
		-lpthread -ldl $(PNGFLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS)

local_laplacian_viz.o: local_laplacian_gen
	HL_TRACE=3 ./local_laplacian_gen 6 0
	mv local_laplacian.o local_laplacian_viz.o

local_laplacian.mp4: process_viz
	bash viz.sh

clean: clean_bench clean_variants
	rm -f out.png process_ref process_auto process_rand process_naive process_sweep \
		local_laplacian local_laplacian*.o process_viz \
		local_laplacian_viz.o local_laplacian_gen \
		local_laplacian.mp4 local_laplacian.h
