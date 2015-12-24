include ../support/Makefile.inc

#CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp -g
CXX=g++ -O3 -march=native -Wall -fopenmp -g

cost_function_test: cost_function_test.cpp
	$(CXX) $(CXXFLAGS) cost_function_test.cpp $(LIB_HALIDE) -o cost_function_test -ldl -lpthread -lz $(LDFLAGS) -ltinfo

cost_function_test_ref: cost_function_test
	./cost_function_test 0

cost_function_test_auto: cost_function_test
	./cost_function_test -1

test_ref: test.cpp cost_function_test_ref
	$(CXX) $(CXXFLAGS) test.cpp cost.o -o test_ref -lpthread -ldl $(PNGFLAGS)

test_auto: test.cpp cost_function_test_auto
	$(CXX) $(CXXFLAGS) test.cpp cost.o -o test_auto -lpthread -ldl $(PNGFLAGS)

bench_ref: test_ref
	rm -rf ref_perf.txt
	for t in 1 2 4 8 ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t ./test_ref >> ref_perf.txt;\
    done

bench_auto: test_auto
	rm -rf auto_perf.txt
	for t in 1 2 4 8 ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t ./test_auto >> auto_perf.txt;\
    done

clean:
	rm -f test cost_function_test cost.h cost.o ref_perf.txt auto_perf.txt test_ref \
    test_auto
