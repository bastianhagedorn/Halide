include ../support/Makefile.inc

#CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp -g
#CXX=g++ -O3 -march=native -Wall -fopenmp -g

.PHONY: clean

mat_mul: mat_mul.cpp
	$(CXX) $(CXXFLAGS) mat_mul.cpp $(LIB_HALIDE) -o mat_mul -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS) -ltinfo

bench_ref: mat_mul
	rm -rf ref_perf.txt
	for t in $(THREADS_TO_TEST) ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t \
		./mat_mul 0 2>> ref_perf.txt;\
    done

bench_auto: mat_mul
	rm -rf auto_perf.txt
	for t in $(THREADS_TO_TEST) ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t \
		./mat_mul -1 2>> auto_perf.txt;\
    done

clean:
	rm -f mat_mul mat_mul.h ref_perf.txt auto_perf.txt
