include ../support/Makefile.inc

#CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp -g
#CXX=g++ -O3 -march=native -Wall -fopenmp -g

data_dependent_test: data_dependent_test.cpp
	$(CXX) $(CXXFLAGS) data_dependent_test.cpp $(LIB_HALIDE) -o data_dependent_test -ldl -lpthread -lz $(LDFLAGS) -ltinfo

data_dependent_test_ref: data_dependent_test
	./data_dependent_test 0

data_dependent_test_auto: data_dependent_test
	./data_dependent_test -1

test_ref: test.cpp data_dependent_test_ref
	$(CXX) $(CXXFLAGS) test.cpp data_dep.o -o test_ref -lpthread -ldl $(PNGFLAGS)

test_auto: test.cpp data_dependent_test_auto
	$(CXX) $(CXXFLAGS) test.cpp data_dep.o -o test_auto -lpthread -ldl $(PNGFLAGS)

bench_ref: test_ref
	rm -rf ref_perf.txt
	for t in $(THREADS_TO_TEST) ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t ./test_ref >> ref_perf.txt;\
    done

bench_auto: test_auto
	rm -rf auto_perf.txt
	for t in $(THREADS_TO_TEST) ; do \
        OMP_NUM_THREADS=$$t HL_NUM_THREADS=$$t ./test_auto >> auto_perf.txt;\
    done

clean:
	rm -f test data_dependent_test data_dep.h data_dep.o ref_perf.txt auto_perf.txt test_ref \
    test_auto
