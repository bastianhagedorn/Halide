include ../support/Makefile.inc

#CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp -g
CXX=g++ -O3 -march=native -Wall -fopenmp -g

.PHONY: clean

conv_bench: conv_bench.cpp
	$(CXX) $(CXXFLAGS) conv_bench.cpp $(LIB_HALIDE) -o conv_bench -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

bench_ref: conv_bench
	rm -rf ref_perf.txt
	for t in 1 2 4 8 ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t \
		./conv_bench 0 2>> ref_perf.txt;\
    done

bench_auto: conv_bench
	rm -rf auto_perf.txt
	for t in 1 2 4 8 ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t \
		./conv_bench -1 2>> auto_perf.txt;\
    done

clean:
	rm -f conv_bench conv_bench.h ref_perf.txt auto_perf.txt
