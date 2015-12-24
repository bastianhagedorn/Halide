include ../support/Makefile.inc

#CXX=icpc -O3 -march=native -ansi-alias -Wall -openmp -g
CXX=g++ -O3 -march=native -Wall -fopenmp -g

overlap_test: overlap_test.cpp
	$(CXX) $(CXXFLAGS) overlap_test.cpp $(LIB_HALIDE) -o overlap_test -ldl -lpthread -lz $(LDFLAGS) -ltinfo

overlap_test_ref: overlap_test
	./overlap_test 0

overlap_test_auto: overlap_test
	./overlap_test -1

test_ref: test.cpp overlap_test_ref
	$(CXX) $(CXXFLAGS) test.cpp overlap.o -o test_ref -lpthread -ldl $(PNGFLAGS)

test_auto: test.cpp overlap_test_auto
	$(CXX) $(CXXFLAGS) test.cpp overlap.o -o test_auto -lpthread -ldl $(PNGFLAGS)

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
	rm -f test overlap_test overlap.h overlap.o ref_perf.txt auto_perf.txt test_ref \
    test_auto
